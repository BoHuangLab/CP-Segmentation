import tifffile

from cellpose import models
from cellpose import plot
from cellpose import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data

import numpy as np
import os
import skimage

import pandas as pd
from skimage.filters import threshold_local
import json

import gc

def get_images_and_metadata(parent_path, image_extension):
    img_paths_list = []
    metadata_paths_list = []
    for root, dirs, files in os.walk(parent_path):
        for file_name in files:
            if file_name.endswith((image_extension)):
                img_name = file_name.split(image_extension)[0]
                img_paths_list.append(root + '/' + file_name)
                metadata_paths_list.append(root + '/' + img_name +
                                           '_metadata.txt')

    return img_paths_list, metadata_paths_list


def import_images_and_metadata(img_paths_list, metadata_paths_list=None):
    img_dict = {}
    metadata_dict = {}

    img_list = [skimage.io.imread(f) for f in img_paths_list]

    if metadata_paths_list != None:
        metadata_list = []
        for i in range(0, len(img_list)):
            with open(metadata_paths_list[i]) as f:
                metadata = json.load(f)
                metadata_summary = metadata['Summary']
                metadata_dict.update({img_paths_list[i]: metadata_summary})

            slices = int(metadata_summary['Slices'])
            num_channels = int(metadata_summary['Channels'])
            height = int(metadata_summary['Height'])
            width = int(metadata_summary['Width'])

            img_list[i] = reshape_image(img_list[i], slices, num_channels, height,
                                        width)
            img_dict.update({img_paths_list[i]: img_list[i]})

    return img_dict, metadata_dict


def sort_channels(nucleus_channels, cyto_channels, slices, image):
    if len(cyto_channels) > 0:
        if slices == 1:
            cyto_channel = image[cyto_channels[0] - 1, :, :]
        elif slices > 1:
            cyto_channel = image[:, cyto_channels[0] - 1, :, :]

    elif len(cyto_channels) == 0:
        cyto_channel = []

    if len(nucleus_channels) > 0:
        if slices == 1:
            nucleus_channel = image[nucleus_channels[0] - 1, :, :]
        elif slices > 1:
            nucleus_channel = image[:, nucleus_channels[0] - 1, :, :]

    elif len(nucleus_channels) == 0:
        nucleus_channel = []

    return cyto_channel.astype('uint16'), nucleus_channel.astype('uint16')


def stack_and_combine_channels(nucleus_channels, cyto_channels, slices, image):
    nucleus_channel_data = []
    cyto_channel_data = []
    if len(cyto_channels) > 0:
        for channel in cyto_channels:
            if slices == 1:
                cyto_channel_data.append(image[channel - 1, :, :])
            elif slices > 1:
                cyto_channel_data.append(image[:, channel - 1, :, :])
            cyto_channel = np.mean(cyto_channel_data, axis=0)
    elif len(cyto_channels) == 0:
        cyto_channel = []

    if len(nucleus_channels) > 0:
        for channel in nucleus_channels:
            if slices == 1:
                nucleus_channel_data.append(image[channel - 1, :, :])
            elif slices > 1:
                nucleus_channel_data.append(image[:, channel - 1, :, :])
            nucleus_channel = np.mean(nucleus_channel_data, axis=0)
    elif len(nucleus_channels) == 0:
        nucleus_channel = []

    return cyto_channel, nucleus_channel


def stack_images(cyto_channel, nucleus_channel, slices):

    if slices == 1:
        stack_axis = 0
    elif slices > 1:
        stack_axis = 1

    if len(cyto_channel) > 0 and len(nucleus_channel) > 0:
        final_image = np.stack([cyto_channel, nucleus_channel],
                               axis=stack_axis)

    elif len(cyto_channel) == 0 and len(nucleus_channel) > 0:
        final_image = nucleus_channel

    elif len(cyto_channel) > 0 and len(nucleus_channel) == 0:
        final_image = cyto_channel

    return final_image.astype('uint16')


def reshape_image(image, slices, num_channels, height, width):
    shape = image.shape

    if slices == 1:
        if shape[0] != num_channels:
            image = image.transpose(2,0,1)

    elif slices > 1:
        if shape[1] != num_channels:
            image = image.transpose(0,3,1,2)

    return image


def prepare_image_to_segment(nucleus_channels,
                             cyto_channels,
                             img_dict,
                             metadata_dict,
                             combine_channels='no'):

    img_path_list = list(img_dict.keys())
    img_list = list(img_dict.values())
    metadata_list = list(metadata_dict.values())

    stripped_image_dict_2d = {}
    stripped_image_dict_3d = {}
    for i in range(0, len(img_list)):

        image = img_list[i]
        path = img_path_list[i]
        slices = int(metadata_dict[path]['Slices'])

        if combine_channels == 'yes':
            cyto_channel, nucleus_channel = stack_and_combine_channels(
                nucleus_channels, cyto_channels, slices, image)

        elif combine_channels == 'no':
            cyto_channel, nucleus_channel = sort_channels(
                nucleus_channels, cyto_channels, slices, image)

        if slices == 1:
            final_image = stack_images(cyto_channel, nucleus_channel, slices)
            stripped_image_dict_2d.update({img_path_list[i]: final_image})

        elif slices > 1:
            final_image = stack_images(cyto_channel, nucleus_channel, slices)
            stripped_image_dict_3d.update({img_path_list[i]: final_image})

    return stripped_image_dict_2d, stripped_image_dict_3d


def select_model_channels(cyto_channels, nucleus_channels):
    if len(cyto_channels) > 0 and len(nucleus_channels) > 0:
        channels = [1, 2]

    elif len(cyto_channels) == 0 or len(nucleus_channels) == 0:
        channels = [0, 0]

    return channels


def move_npy_to_folder(current_folder, destination_folder):
    for root, dirs, files in os.walk(current_folder):
        for name in files:
            dirs[:] = []
            if name.endswith((".npy")):
                new_destination = destination_folder + '/' + name
                os.replace(root + '/' + name, new_destination)
    return new_destination

def d_plot(image,mask,export_path):
    mask_array = np.array(mask)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    A = np.array([i for i in range(mask_array.shape[1])])
    B =np.array([[i] for i in range(mask_array.shape[2])])

    from matplotlib.colors import LinearSegmentedColormap

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
    gray_array = plt.get_cmap('gray')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,.3,ncolors)
    gray_array[:,-1] = np.linspace(0.0,1.0,ncolors)



    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    map_object2 = LinearSegmentedColormap.from_list(name='gray_alpha',colors=gray_array)



    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    plt.register_cmap(cmap=map_object2)



    AB = A*B*0

    for i in range(mask_array.shape[0]):
        C =mask_array[i,:,:].transpose(1,0)
        Z = AB+i
        D= image[i,:,:,0].transpose(1,0)*(C>0)

        scamap2 = plt.cm.ScalarMappable(cmap='gray_alpha')
        fcolors2 = scamap2.to_rgba(D)
        ax.plot_surface(A, B, Z, facecolors=fcolors2, cmap='gray_alpha')

    fig = plt.gcf()
    fig.savefig(export_path + '.svg')

    for i in range(mask_array.shape[0]):
        C =mask_array[i,:,:].transpose(1,0)
        Z = AB+i
        scamap = plt.cm.ScalarMappable(cmap='rainbow_alpha')
        fcolors = scamap.to_rgba(C)


        ax.plot_surface(A, B, Z, facecolors=fcolors, cmap='rainbow_alpha')


    fig = plt.gcf()
    fig.savefig(export_path + 'Color Coded.svg')
    plt.show()

def segmentation_model(og_image_dict,
                       final_image_dict,
                       metadata_dict,
                       estimated_diameter,
                       channels,
                       nucleus_channels,
                       cyto_channels,
                       feature_to_segment,
                       image_extension):

    og_image_list = []
    image_paths = list(final_image_dict.keys())
    image_list = list(final_image_dict.values())

    for key in image_paths:
        og_image_list.append(og_image_dict[key])

    model = models.Cellpose(model_type=feature_to_segment)


    if feature_to_segment == 'nuclei' and channels != [0, 0]:
        for idx in range(len(image_list)):
            image_list[idx] = image_list[idx][1]
        channels = [0, 0]
    #Run segmentation model
    masks, flows, styles, diams = model.eval(image_list,
                                             diameter=estimated_diameter,
                                             flow_threshold=None,
                                             channels=channels)

    io.masks_flows_to_seg(og_image_list, masks, flows, diams, image_paths,
                          channels)
    print('Exporting mask plots...')

    nimg = len(image_list)
    for idx in range(nimg):
        maski = masks[idx]
        flowi = flows[idx][0]
        original_image = og_image_list[idx]
        fig = plt.figure(figsize=(16, 6.66))

        path = image_paths[idx]
        active_folder = os.path.split(path)[0]
        filename = os.path.split(path)[1]
        new_folder = active_folder + '/' + feature_to_segment + '_mask'

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        npy_path = move_npy_to_folder(active_folder, new_folder)

        name = filename.replace(image_extension, '')
        title = name + ' - ' + feature_to_segment

        fig.suptitle(title)


        plot.show_segmentation(fig,
                               image_list[idx],
                               maski,
                               flowi,
                               channels=channels)
        plt.tight_layout()

        plt.savefig(new_folder + '/' + title + '.svg')
        np.save(new_folder + '/masks.npy',maski)
        np.save(new_folder + '/flows.npy',flowi)

        print('Analyzing ' + str(name) + 'mask data...')

        img_metadata_dict = metadata_dict[path]

        analyze_masks(nucleus_channels, cyto_channels, original_image,
                      img_metadata_dict, title, npy_path, new_folder)

        og_image_list[idx] = 0
        masks[idx] = 0
        flows[idx] =  0
        image_paths[idx] = 0
        gc.collect()

def segmentation_model_3d(og_image_dict,
                       final_image_dict,
                       metadata_dict,
                       estimated_diameter,
                       channels,
                       nucleus_channels,
                       cyto_channels,
                       feature_to_segment,
                       image_extension):

    og_image_list = []
    image_paths = list(final_image_dict.keys())
    image_list = list(final_image_dict.values())

    for key in image_paths:
        og_image_list.append(og_image_dict[key])

    model = models.Cellpose(model_type=feature_to_segment)

    if estimated_diameter == None:
            print('You must input an estimated diameter \
            to perform 3D segmentation. \n \
            Only 2D segmentation will be performed.')

    if feature_to_segment == 'nuclei' and channels != [0, 0]:
        for idx in range(len(image_list)):
            image_list[idx] = image_list[idx][:,1,:,:]
            print(image_list[idx].shape)
        channels = [0, 0]
    #Run segmentation model
    masks, flows, styles, diams = model.eval(image_list,
                                             diameter=estimated_diameter,
                                             flow_threshold=None,
                                             channels=channels, do_3D = True)


    print('Exporting mask plots...')


    nimg = len(image_list)
    print(nimg)
    for idx in range(nimg):
        original_image = og_image_list[idx]
        maski = masks[idx]
        flowi = flows[idx]
        path = image_paths[idx]
        active_folder = os.path.split(path)[0]
        filename = os.path.split(path)[1]
        new_folder = active_folder + '/' + feature_to_segment + '_mask'

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        name = filename.replace(image_extension, '')
        title = name + ' - ' + feature_to_segment



        d_plot(image_list[idx], maski,new_folder + '/' + title)

        np.save(new_folder +'/masks.npy',maski)
        np.save(new_folder+'/flows.npy',flowi)


        print('Analyzing ' + str(name) + 'mask data...')

        img_metadata_dict = metadata_dict[path]

        analyze_masks_3d(nucleus_channels, cyto_channels, original_image,
                      img_metadata_dict, title, maski,flows,diams, new_folder)

        og_image_list[idx] = 0
        masks[idx] = 0
        flows[idx] =  0
        image_paths[idx] = 0
        gc.collect()

def rolling_ball(image, radius=50, light_bg=False):
        from skimage.morphology import white_tophat, black_tophat, disk
        str_el = disk(radius)
        if light_bg:
            return black_tophat(image, str_el)
        else:
            return white_tophat(image, str_el)

def generate_channel_data(image_data, block_size,
                          export_path):

    bg_subtracted_array = rolling_ball(image_data,radius=block_size/2)

    background = image_data - bg_subtracted_array

    return bg_subtracted_array, background


def calculate_channel_stats(mask_name_array, bg_subtracted_array, bg_array):

    intensity = np.sum((mask_name_array*bg_subtracted_array).flatten())
    variance = np.var((mask_name_array*bg_subtracted_array).flatten())

    background_intensity = np.sum((mask_name_array*bg_array).flatten())

    return intensity, variance, background_intensity


def analyze_masks(nucleus_channels, cyto_channels, original_image,
                  metadata_dict, title, npy_path, export_path):

    gc.collect()

    channel_names = list(metadata_dict['ChNames'])
    channels_to_analyze = nucleus_channels + cyto_channels

    data = np.load(npy_path, allow_pickle=True).item()

    outlines_array = data['outlines']
    masks_array = data['masks']
    flows_array = data['flows']

    block_size = int(data['est_diam'].item())

    if block_size % 2 == 0:
        block_size += 1

    last_mask = np.max(data['masks'])
    full_columns = ['Mask Number', 'Center X', 'Center Y', 'Area (px^2)']
    intensity_df = pd.DataFrame(columns=full_columns)

    channel_intensity_dict = {}

    print('Subtracting Background...')

    background_list = []
    background_subtracted_list = []
    for channel in channels_to_analyze:
        channel_label = str(channel_names[channel-1])
        full_columns.extend([
            channel_label + ' Intensity',
            channel_label + ' Variance',
            channel_label + ' Background Intensity'
        ])


        image_data = original_image[channel-1]

        bg_subtracted_array, bg_array = generate_channel_data(image_data, block_size, export_path)

        channel_intensity_dict.update(
                {channel_label: [bg_subtracted_array[channel], bg_array[channel]]})


        background_list.append(bg_array)
        background_subtracted_list.append(bg_subtracted_array)

    background_list = np.dstack(background_list)
    background_subtracted_list = np.dstack(background_subtracted_list)

    #background_list = bg_array
    #background_subtracted_list = bg_subtracted_array

    tifffile.imwrite(export_path + '/Background.tif', background_list, imagej=True)

    tifffile.imwrite(export_path + '/Background Subtracted Image.tif', background_subtracted_list, imagej=True)




    print('Calculating Mask Data..')

    for mask_name in range(1, last_mask + 1):

        new_data_dict = {}

        mask_name_array = masks_array==mask_name

        new_data_dict.update({'Mask Number': mask_name})

        area = np.sum(1*mask_name_array)
        new_data_dict.update({'Area (px^2)': area})

        if area > 0:
            nonzero_array = np.nonzero(mask_name_array)


            x_min =  np.min(nonzero_array[1]) + 1
            x_max = np.max(nonzero_array[1]) + 1
            x_center = (x_min + x_max) / 2
            new_data_dict.update({'Center X': x_center})

            y_min = np.min(nonzero_array[0]) + 1
            y_max = np.max(nonzero_array[0]) + 1
            y_center = (y_min + y_max) / 2
            new_data_dict.update({'Center Y': y_center})

            for channel_label in channel_names:
                if channel_label in list(channel_intensity_dict.keys()):

                    bg_subtracted_array = channel_intensity_dict[channel_label][0]
                    bg_array = channel_intensity_dict[channel_label][1]
                    intensity, variance, background_intensity = calculate_channel_stats(
                        mask_name_array, bg_subtracted_array, bg_array)

                    new_data_dict.update({
                        channel_label + ' Intensity':
                        intensity,
                        channel_label + ' Variance':
                        variance,
                        channel_label + ' Background Intensity':
                        background_intensity
                    })

        new_df = pd.DataFrame(new_data_dict, index=[mask_name-1])

        intensity_df = intensity_df.append(new_df, ignore_index=True)

    first_col = intensity_df.pop('Mask Number')
    intensity_df.insert(0, 'Mask Number', first_col)
    intensity_df.to_csv(export_path + '/' + title +
                        ' Cell Intensity Values.csv',
                        index=False)


def analyze_masks_3d(nucleus_channels, cyto_channels, original_image,
                  metadata_dict, title, masks,flows,diams, export_path):

    channel_names = list(metadata_dict['ChNames'])
    channels_to_analyze = nucleus_channels + cyto_channels


    masks_array = np.array(masks)

    flows_array = flows

    block_size = int(diams)


    if block_size % 2 == 0:
        block_size += 1

    last_mask = np.max(masks)
    full_columns = ['Mask Number', 'Center X', 'Center Y', 'Center Z', 'Volume (px^3)']


    channel_intensity_dict = {}

    print('Subtracting Background...')

    background_images = []
    background_subtracted_images = []

    for channel in channels_to_analyze:

        channel_images = []
        channel_subtracted_images = []

        channel_label = str(channel_names[channel-1])
        full_columns.extend([
            channel_label + ' Intensity',
            channel_label + ' Variance',
            channel_label + ' Background Intensity'
        ])

        image_data = original_image[:,channel-1,:,:]

        print('Calculating Mask Data..')

        layer_intensity_dict = {}

        for layer in range(len(image_data)):


            bg_subtracted_array, bg_array = generate_channel_data(image_data[layer], block_size, export_path)

            layer_intensity_dict.update(
                {layer:[bg_subtracted_array, bg_array]})

            channel_images.append(bg_array)
            channel_subtracted_images.append(bg_subtracted_array)

        channel_intensity_dict.update({channel_label:layer_intensity_dict})
        background_images.append(np.stack(channel_images))
        background_subtracted_images.append(np.stack(channel_subtracted_images))
        intensity_df = pd.DataFrame(columns=full_columns)


    for mask_name in range(1, last_mask + 1):

        gc.collect()

        new_data_dict = {}

        mask_name_array = np.array(masks_array)==mask_name

        new_data_dict.update({'Mask Number': mask_name})

        area = np.sum((mask_name_array))
        new_data_dict.update({'Volume (px^3)': area})

        if area > 0:
            nonzero_array = np.nonzero(mask_name_array)

            x_min =  np.min(nonzero_array[1]) + 1
            x_max = np.max(nonzero_array[1]) + 1
            x_center = (x_min + x_max) / 2
            new_data_dict.update({'Center X': x_center})

            y_min = np.min(nonzero_array[2]) + 1
            y_max = np.max(nonzero_array[2]) + 1
            y_center = (y_min + y_max) / 2
            new_data_dict.update({'Center Y': y_center})

            z_min = np.min(nonzero_array[0]) + 1
            z_max = np.max(nonzero_array[0]) + 1
            z_center = (z_min + z_max) / 2
            new_data_dict.update({'Center Z': z_center})

            for channel_label in channel_names:
                if channel_label in list(channel_intensity_dict.keys()):

                    total_intensity = 0
                    total_background_intensity = 0
                    total_variance = 0

                    for layer in range(len(image_data)):

                        bg_subtracted_array = channel_intensity_dict[channel_label][layer][0]
                        bg_array = channel_intensity_dict[channel_label][layer][1]
                        intensity, variance, background_intensity = calculate_channel_stats(
                            mask_name_array[layer], bg_subtracted_array, bg_array)

                        total_intensity += intensity
                        total_background_intensity += background_intensity
                        total_variance += variance

                    new_data_dict.update({
                        channel_label + ' Intensity':
                        total_intensity,
                        channel_label + ' Variance':
                        total_variance,
                        channel_label + ' Background Intensity':
                        total_background_intensity
                    })

            new_df = pd.DataFrame(new_data_dict, index=[mask_name-1])

            intensity_df = intensity_df.append(new_df, ignore_index=True)

        tifffile.imwrite(export_path + '/Background Stack.tif', np.stack(background_images).astype('uint16'), imagej=True)

        tifffile.imwrite(export_path +
                             '/Background Subtracted Image Stack.tif', np.stack(background_subtracted_images).astype('uint16'), imagej=True)





        first_col = intensity_df.pop('Mask Number')
        intensity_df.insert(0, 'Mask Number', first_col)
        intensity_df.to_csv(export_path + '/' + title +
                            ' Cell Volume Intensity Values.csv',
                            index=False)



def run_segmentation(parent_path, image_extension, nucleus_channels,
                     cyto_channels, combine_channels, feature_to_segment,
                     estimated_diameter):

    channels = select_model_channels(cyto_channels, nucleus_channels)

    print("Finding images...")

    img_paths_list, metadata_paths_list = get_images_and_metadata(
        parent_path, image_extension)

    print("Sorting image dictinonaries...")

    img_dict, metadata_dict = import_images_and_metadata(
        img_paths_list, metadata_paths_list)

    print("Combining channels for segmentation...")

    image_dict_2d, image_dict_3d = prepare_image_to_segment(
        nucleus_channels, cyto_channels, img_dict, metadata_dict,
        combine_channels)

    print("Segmenting Images...")

    if len(image_dict_2d) > 0:
        if feature_to_segment == 'both':
            segmentation_model(img_dict,
                               image_dict_2d,
                               metadata_dict,
                               estimated_diameter,
                               channels,
                               nucleus_channels,
                               cyto_channels,
                               'cyto',
                               image_extension)


            segmentation_model(img_dict,
                               image_dict_2d,
                               metadata_dict,
                               estimated_diameter,
                               channels,
                               nucleus_channels,
                               cyto_channels,
                               'nuclei',
                               image_extension)

        elif feature_to_segment == 'cyto' or feature_to_segment == 'nuclei':
            segmentation_model(img_dict,
                               image_dict_2d,
                               metadata_dict,
                               estimated_diameter,
                               channels,
                               nucleus_channels,
                               cyto_channels,
                               feature_to_segment,
                               image_extension)

    if len(image_dict_3d) > 0:
        if feature_to_segment == 'both':
            segmentation_model(img_dict,
                               image_dict_3d,
                               metadata_dict,
                               estimated_diameter,
                               channels,
                               nucleus_channels,
                               cyto_channels,
                               'cyto',
                               image_extension)


            segmentation_model_3d(img_dict,
                               image_dict_3d,
                               metadata_dict,
                               estimated_diameter,
                               channels,
                               nucleus_channels,
                               cyto_channels,
                               'nuclei',
                               image_extension)

        elif feature_to_segment == 'cyto' or feature_to_segment == 'nuclei':
            segmentation_model_3d(img_dict,
                               image_dict_3d,
                               metadata_dict,
                               estimated_diameter,
                               channels,
                               nucleus_channels,
                               cyto_channels,
                               feature_to_segment,
                               image_extension)

    gc.collect()

    print("------DONE------")
