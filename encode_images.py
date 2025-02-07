import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import shutil
import errno

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_ref_images_dirs(path, extra_path, images, path_list, recursive=True):
    for x in os.listdir(path):
        raw_dir = os.path.join(path, x)
        if os.path.isdir(raw_dir):
            if recursive:
                images, path_list = get_ref_images_dirs(raw_dir, os.path.join(extra_path, x), images, path_list)
        if x.endswith(".png"):
            images.append(os.path.join(path, x))
            path_list.append(extra_path)
    return images, path_list



def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')

    parser.add_argument('--network_pkl', default='gdrive:networks/stylegan2-ffhq-config-f.pkl', help='Path to local copy of stylegan2-ffhq-config-f.pkl')

    # for now it's unclear if larger batch leads to better performance/quality
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=1., help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    ref_images, ref_paths = get_ref_images_dirs(args.src_dir, "", [], [])
    #ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir) if x.endswith(".png")]
    #ref_images = list(filter(os.path.isfile, ref_images))
    if os.path.exists(os.path.join(args.src_dir, "crops.pickle")):
        shutil.copy(os.path.join(args.src_dir, "crops.pickle"), args.dlatent_dir)

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks(args.network_pkl)

    batch_size = args.batch_size
    generator = Generator(Gs_network, batch_size, randomize_noise=args.randomize_noise)
    perceptual_model = PerceptualModel(args.image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    image_loss = None
    try:
        image_loss = pickle.load(open(os.path.join(args.dlatent_dir, "image_losses.pickle"), "rb"))
    except:
        image_loss = {}

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch, path_batch in zip(tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size),
                            tqdm(split_to_batches(ref_paths, batch_size), total=len(ref_paths)//batch_size)):

        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        # Planteado solo para batches de 1
        contains = False
        for name in names:
            if name in image_loss.keys():
                contains = True
                break

        if contains:
            continue

        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=100, learning_rate=args.lr)
        pbar = tqdm(op, leave=False, total=100)
        print("IMAGE NAME", ' '.join(names))
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
        print("100 loss: ", loss)
        if loss > 1.3:
            for name in names:
                image_loss[name] = {"loss": 1000, "loss_100": loss, "latent": None, "generated": None}

            pickle.dump(image_loss, open(os.path.join(args.dlatent_dir, "image_losses.pickle"), "wb"))
            continue

        loss_100 = loss

        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations-100, learning_rate=args.lr)
        pbar = tqdm(op, leave=False, total=args.iterations - 100)
        for loss in pbar:
            pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)

        print(' '.join(names), ' loss:', loss)
        #if loss > 1.25:
        #    continue
        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name, path in zip(generated_images, generated_dlatents, names, path_batch):
            img = PIL.Image.fromarray(img_array, 'RGB')
            g_images_dir = os.path.join(args.generated_images_dir, path)
            latents_dir = os.path.join(args.dlatent_dir, path)
            print("IMAGES_DIR", g_images_dir)
            os.makedirs(g_images_dir, exist_ok=True)
            os.makedirs(latents_dir, exist_ok=True)
            img.save(os.path.join(g_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(latents_dir, f'{img_name}.npy'), dlatent)

            image_loss[img_name] = {"loss": loss, "loss_100": loss_100, "latent": os.path.join(latents_dir, f'{img_name}.npy'),
                                    "generated": os.path.join(g_images_dir, f'{img_name}.png')}
        
        pickle.dump(image_loss, open(os.path.join(args.dlatent_dir, "image_losses.pickle"), "wb"))

        generator.reset_dlatents()


if __name__ == "__main__":
    main()
