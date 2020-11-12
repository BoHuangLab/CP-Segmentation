import os
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2gray
from skimage.morphology import white_tophat, black_tophat, disk

import tifffile

from cellpose import models
from cellpose import plot
from cellpose import io

def run_segmentation(parent_folder, images,metadata, image_paths, cyto_channels, nucleus_channels, estimated_diameter, model,feature_to_segment):

    channels = select_model_channels(cyto_channels, nucleus_channels)

    print("Combining channels for segmentation...")

    images_2d, images_3d = prepare_image_to_segment(images, metadata, image_paths, cyto_channels, nucleus_channels)

    print("Segmenting Images...")

    if len(images_2d) > 0:
        if feature_to_segment == 'both':
            segmentation_model(images_2d,channels,cyto_channels, nucleus_channels, estimated_diameter,
                               'cyto', model)


            segmentation_model(images_2d,channels,cyto_channels, nucleus_channels, estimated_diameter,
                               'nuclei',model)

        elif feature_to_segment in ('cyto','nuclei'):
            segmentation_model(images_2d,channels,cyto_channels, nucleus_channels, estimated_diameter,feature_to_segment,model)

    if len(images_3d) > 0:
        if feature_to_segment == 'both':
            segmentation_model_3d(images_3d,channels,cyto_channels, nucleus_channels, estimated_diameter, 'cyto',model)


            segmentation_model_3d(images_3d,channels,cyto_channels, nucleus_channels, estimated_diameter, 'nuclei',model)

        elif feature_to_segment in ('cyto','nuclei'):
            segmentation_model_3d(images_3d,channels,cyto_channels, nucleus_channels, estimated_diameter,feature_to_segment,model)

    gc.collect()

    print("------DONE------")

def select_model_channels(cyto_channels, nucleus_channels):
    if len(cyto_channels) > 0 and len(nucleus_channels) > 0:
        channels = [1, 2]

    elif len(cyto_channels) == 0 or len(nucleus_channels) == 0:
        channels = [0, 0]

    return channels

def prepare_image_to_segment(images, metadata, image_paths, cyto_channels, nucleus_channels):

    images_2d = []
    images_3d = []
    for idx, image in enumerate(images):

        data_type = image.dtype

        slices = int(metadata[idx]['SizeZ'])

        cyto_channel, nucleus_channel = stack_and_combine_channels(image, slices, cyto_channels, nucleus_channels)

        if slices == 1:
            final_image = stack_images(cyto_channel, nucleus_channel, slices).astype(data_type)
            images_2d.append([image, final_image, metadata[idx],image_paths[idx]])

        elif slices > 1:
            final_image = stack_images(cyto_channel, nucleus_channel, slices).astype(data_type)
            images_3d.append([image, final_image, metadata[idx],image_paths[idx]])

    return images_2d, images_3d


def stack_and_combine_channels(image, slices, cyto_channels, nucleus_channels):

    nucleus_channel_data = []
    cyto_channel_data = []

    if len(cyto_channels) > 0:
        for channel in cyto_channels:
            if slices == 1:
                cyto_channel_data.append(image[channel - 1, :, :])
            elif slices > 1:
                cyto_channel_data.append(image[:, channel - 1, :, :])
            cyto_channel = np.sum(cyto_channel_data, axis=0)

    elif len(cyto_channels) == 0:
        cyto_channel = []

    if len(nucleus_channels) > 0:
        for channel in nucleus_channels:
            if slices == 1:
                nucleus_channel_data.append(image[channel - 1, :, :])
            elif slices > 1:
                nucleus_channel_data.append(image[:, channel - 1, :, :])
            nucleus_channel = np.sum(nucleus_channel_data, axis=0)

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

    return final_image

def segmentation_model(image_group,channels,cyto_channels, nucleus_channels, estimated_diameter,
                   feature_to_segment, model):

    original_images = [set[0] for set in image_group]
    images_to_segment = [set[1] for set in image_group]
    metadata = [set[2] for set in image_group]
    image_paths = [set[3] for set in image_group]

    if feature_to_segment == 'nuclei' and channels != [0, 0]:
        for idx, image in enumerate(images_to_segment):
            images_to_segment[idx] = image[1]
        channels = [0, 0]


    #Run segmentation model
    masks, flows, styles, diams = model.eval(images_to_segment,
                                             diameter=estimated_diameter,
                                             flow_threshold=None,
                                             channels=channels)

    io.masks_flows_to_seg(original_images, masks, flows, diams, image_paths,
                          channels)

    print('Exporting mask plots...')

    for idx, image in enumerate(original_images):
        path = image_paths[idx]

        active_folder = os.path.split(path)[0]
        new_folder = os.path.join(active_folder,str(feature_to_segment)+'_mask')

        name = Path(path).stem

        if '.ome.tif' in path:
            name = name[:-4]

        title = name + ' - ' + feature_to_segment


        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        npy_path = move_npy_to_folder(active_folder, new_folder)

        maski = masks[idx]
        flowi = flows[idx][0]

        fig = plt.figure(figsize=(16, 6.66))

        fig.suptitle(title)

        if image.dtype is not 'uint8':
            plot_image = (image/np.max(image)*255).astype('uint8')

        plot.show_segmentation(fig,
                               plot_image,
                               maski,
                               flowi,
                               channels=channels)
        plt.tight_layout()

        plt.savefig(os.path.join(new_folder,title+'.svg'))

        tifffile.imwrite(new_folder + '/masks.tif',maski.astype('uint16'))

        tifffile.imwrite(new_folder + '/flows.tif',flowi)

        print('Analyzing ' + str(name) + 'mask data...')

        analyze_masks(image, metadata[idx],cyto_channels,nucleus_channels,title,npy_path,new_folder)

        #setting these to 0 to clear up some memoery
        original_images[idx] = 0
        images_to_segment[idx] = 0
        metadata[idx] = 0
        image_paths[idx] = 0
        gc.collect()

def move_npy_to_folder(current_folder, destination_folder):
    for root, dirs, files in os.walk(current_folder):
        for name in files:
            dirs[:] = []
            if name.endswith((".npy")):
                new_destination = os.path.join(destination_folder,name)
                os.replace(os.path.join(root,name), new_destination)
    return new_destination

def segmentation_model_3d(image_group,channels,cyto_channels, nucleus_channels, estimated_diameter,
                   feature_to_segment, model):

    original_images = [set[0] for set in image_group]
    images_to_segment = [set[1] for set in image_group]
    metadata = [set[2] for set in image_group]
    image_paths = [set[1] for set in image_group]

    if estimated_diameter is None:
            raise ValueError('You must input an estimated diameter \
            to perform 3D segmentation. \n \
            Only 2D segmentation will be performed.')

    if feature_to_segment == 'nuclei' and channels != [0, 0]:
        for idx, image in enumerate(images_to_segment):
            images_to_segment[idx] = image[idx][:,1,:,:]
        channels = [0, 0]
    #Run segmentation model
    masks, flows, styles, diams = model.eval(images_to_segment,
                                             diameter=estimated_diameter,
                                             flow_threshold=None,
                                             channels=channels, do_3D = True)


    print('Exporting mask plots...')


    for idx, image in enumerate(original_images):
        path = image_paths[idx]

        active_folder = os.path.split(path)[0]
        new_folder = os.path.join(active_folder,str(feature_to_segment)+'_mask')

        if '.ome.tif' in path:
            name = name[:-4]
        else:
            name = Path(path).stem
        title = name + ' - ' + feature_to_segment

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        npy_path = move_npy_to_folder(active_folder, new_folder)

        maski = masks[idx]
        flowi = flows[idx][0]
        diami = diams[idx]

        d_plot(image, maski,new_folder + '/' + title)

        tifffile.imwrite(new_folder + '/masks.tif',maski)

        tifffile.imwrite(new_folder + '/flows.tif',flowi)


        print('Analyzing ' + str(name) + 'mask data...')

        analyze_masks_3d(image, metadata[idx],cyto_channels,nucleus_channels,title,npy_path,new_folder,maski,diami)

        #setting these to 0 to clear up some memoery
        original_images[idx] = 0
        images_to_segment[idx] = 0
        metadata[idx] = 0
        image_paths[idx] = 0
        gc.collect()

def analyze_masks(image, metadata,cyto_channels,nucleus_channels,title,npy_path,export_path):

    gc.collect()

    channel_names = metadata['Channels']

    if len(channel_names) != metadata['SizeC']:
        channel_names = ['Channel ' + str(i+1) for i in range(int(metadata['SizeC']))]

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

    for channel in range(len(metadata['Channels'])):

        channel_label = str(channel_names[channel])
        full_columns.extend([
            channel_label + ' Intensity (Magnitude/px^2)',
            channel_label + ' Standard Deviation (Magnitude/px^2)',
            channel_label + ' Background Intensity (Magnitude/px^2)'
        ])


        image_data = image[channel]

        bg_subtracted_array, bg_array = generate_channel_data(image_data, block_size*3, export_path)

        channel_intensity_dict.update(
                {channel_label: [bg_subtracted_array[channel], bg_array[channel]]})


        background_list.append(bg_array)
        background_subtracted_list.append(bg_subtracted_array)

    background_list = np.stack(background_list, axis = 0)
    background_subtracted_list = np.stack(background_subtracted_list, axis = 0)

    #background_list = bg_array
    #background_subtracted_list = bg_subtracted_array

    tifffile.imwrite(os.path.join(export_path,'Background.tif'), background_list, imagej=True)

    tifffile.imwrite(os.path.join(export_path,'Background Subtracted Image.tif'), background_subtracted_list, imagej=True)

    print('Calculating Mask Data..')

    for mask_name in range(1, last_mask + 1):

        new_data_dict = {}

        mask_name_array = masks_array==mask_name

        new_data_dict.update({'Mask Number': mask_name})

        area = np.sum(1*mask_name_array)
        new_data_dict.update({'Area (px^2)': area})

        if area > 0:
            nonzero_array = np.nonzero(mask_name_array)

            flows_name_array = mask_name_array*rgb2gray(flows_array[0][0])

            center = flows_name_array*(flows_name_array < 15)
            center = flows_name_array*(flows_name_array > 0)

            x_center = np.median(center[1]) + 1
            new_data_dict.update({'Center X': x_center})

            y_center = np.median(center[0]) + 1
            new_data_dict.update({'Center Y': y_center})

            for channel_label in channel_names:
                if channel_label in list(channel_intensity_dict.keys()):

                    bg_subtracted_array = channel_intensity_dict[channel_label][0]
                    bg_array = channel_intensity_dict[channel_label][1]
                    intensity, variance, background_intensity = calculate_channel_stats(
                        mask_name_array, bg_subtracted_array, bg_array)

                    new_data_dict.update({
                        channel_label + ' Intensity (Magnitude/px^2)':
                        intensity/area,
                        channel_label + ' Standard Deviation (Magnitude/px^2)':
                        np.sqrt(variance) * (1/area),
                        channel_label + ' Background Intensity (Magnitude/px^2)':
                        background_intensity/area
                    })

        new_df = pd.DataFrame(new_data_dict, index=[mask_name-1])

        intensity_df = intensity_df.append(new_df, ignore_index=True, sort = False)

    first_col = intensity_df.pop('Mask Number')
    intensity_df.insert(0, 'Mask Number', first_col)
    intensity_df.to_csv(os.path.join(export_path,title +
                        ' Cell Intensity Values.csv'),
                        index=False)

def generate_channel_data(image_data, block_size,export_path):

    bg_subtracted_array = rolling_ball(image_data,radius=block_size/2)

    background = image_data - bg_subtracted_array

    return bg_subtracted_array, background

def rolling_ball(image, radius=50, light_bg=False):
    str_el = disk(radius)
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)

def calculate_channel_stats(mask_name_array, bg_subtracted_array, bg_array):

    mask_pixels = (mask_name_array*bg_subtracted_array).flatten()
    mask_pixels = mask_pixels[mask_pixels>0]

    background_pixels = (mask_name_array*bg_subtracted_array).flatten()
    background_pixels = background_pixels[background_pixels>0]

    intensity = np.sum(mask_pixels)
    variance = np.var(mask_pixels)

    background_intensity = np.sum(background_pixels)

    return intensity, variance, background_intensity

def analyze_masks_3d(image, metadata,cyto_channels,nucleus_channels,title,npy_path,export_path,masks,diams):

    gc.collect()

    channel_names = metadata['Channels']

    if len(channel_names) != metadata['SizeC']:
        channel_names = ['Channel ' + str(i+1) for i in range(int(metadata['SizeC']))]

    #channels_to_analyze = range(1,int(metadata['SizeC'])+1)

    masks_array = np.array(masks)

    block_size = int(diams)

    if block_size % 2 == 0:
        block_size += 1

    last_mask = np.max(masks)
    full_columns = ['Mask Number', 'Center X', 'Center Y', 'Center Z', 'Volume (px^3)']


    channel_intensity_dict = {}

    print('Subtracting Background...')

    background_images = []
    background_subtracted_images = []

    for channel in range(len(metadata['Channels'])):

        channel_images = []
        channel_subtracted_images = []

        channel_label = str(channel_names[channel])
        full_columns.extend([
            channel_label + ' Intensity (Magnitude/px^3)',
            channel_label + ' Standard Deviation (Magnitude/px^3)',
            channel_label + ' Background Intensity (Magnitude/px^3)'
        ])

        image_data = image[:,channel,:,:]

        print('Calculating Mask Data..')

        layer_intensity_dict = {}

        for layer in range(len(image_data)):


            bg_subtracted_array, bg_array = generate_channel_data(image_data[layer], block_size*3, export_path)

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

            #need to update this to use flows

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
                        channel_label + ' Intensity (Magnitude/px^3)':
                        total_intensity/area,
                        channel_label + ' Standard Deviation (Magnitude/px^3)':
                        np.sqrt(total_variance) * (1/area),
                        channel_label + ' Background Intensity (Magnitude/px^3)':
                        total_background_intensity/area
                    })

            new_df = pd.DataFrame(new_data_dict, index=[mask_name-1])

            intensity_df = intensity_df.append(new_df, ignore_index=True, sort = False)

        tifffile.imwrite(os.path.join(export_path,'Background Stack.tif'), np.stack(background_images, axis = 0), imagej=True)

        tifffile.imwrite(os.path.join(export_path,'Background Subtracted Image Stack.tif'), np.stack(background_subtracted_images), imagej=True)

        first_col = intensity_df.pop('Mask Number')
        intensity_df.insert(0, 'Mask Number', first_col)

        intensity_df.to_csv(os.path.join(export_path, title +
                            ' Cell Volume Intensity Values.csv'),
                            index=False)

def d_plot(image,mask,export_path):
    mask_array = np.array(mask)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    A = np.array([i for i in range(mask_array.shape[1])])
    B =np.array([[i] for i in range(mask_array.shape[2])])

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
    fig.savefig(os.path.join(export_path,'Gray Stack.svg'))

    for i in range(mask_array.shape[0]):
        C =mask_array[i,:,:].transpose(1,0)
        Z = AB+i
        scamap = plt.cm.ScalarMappable(cmap='rainbow_alpha')
        fcolors = scamap.to_rgba(C)


        ax.plot_surface(A, B, Z, facecolors=fcolors, cmap='rainbow_alpha')


    fig = plt.gcf()
    fig.savefig(os.path.join(export_path,'Color Coded Stack.svg'))
    plt.show()
