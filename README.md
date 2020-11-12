# CP-Segmentation
### Version 2.0.3
This repo contains files used to perform automated cell segmentation via the Cellpose package from mouseland.

## Getting Started
The program recognizes the following image formats: `.ome.tif, .czi, .jpg, .png, .gif, .tif`.

Clone this repo into your own folder into
Open `Segmentation Notebook.ipynb` and fill in the following parameters within the first cell:
* `PARENT_PATH`
  * This should be the highest level folder containing only the images you would like to segment.
* `METADATA_PATH`
  * Path to metadata file, details are specified below in the Metadata section.
  * Default: None
* `FEATURE_TO_SEGMENT`
  *'nuclei' or 'cyto'
* ` NUCLEI_CHANNELS`
  * Enter the channels which correspond to the nucleus.
  * ONLY ENTER THE CHANNELS YOU WOULD LIKE TO BE CONSIDERED IN THE SEGMENTATION.
  * The pixel values from the generated masks will be calculated on all channels.
  * Example: [1,2] would mean nuclei data is on the first and second image channels
 * `CYTO_CHANNELS`
  * Similar to `NUCLEI_CHANNELS`
 * `ESTIMATED_DIAMATER`
  * Enter the cell diameter in terms of pixels.
  * This is not a necessary parameter, but it will be significantly faster and more accurate if a value is offered.
  * REQUIRED for Z-stack images
  * Default: None

## Metadata
Metadata is necessary for calculating pixel values.
OMEXML, the chosen format of The Open Microscopy Environment is used https://docs.openmicroscopy.org/ome-files-cpp/0.5.0/ome-model/manual/html/developers/using-ome-xml.html.

For `.ome.tif` and `.czi`, these are embedded within the image file. You may choose a metadata path if you would like to override the information embedded within the image's metadata (such as channel names).

For all other file types, a metadata path in the OMEXML format is necessary. The most important fields are `SizeX`,`SizeY`,`SizeZ`,`SizeC` and the `Channel..Name` fields. An example XML file is included.

## Output
While running the notebook, plots will be generated in the notebook output. As you see them, note the accuracy of the segmentations. If results are very wrong, try updating the `ESTIMATED_DIAMETER` parameter.

Once run, a new folder will be generated in the same path as the image. The name will be either `cyto_mask` or `nuclei_mask` depending on the type of segmentation you selected.

Within this folder, the following files will be generated:
* `Background Subtracted Image.tif`
  * Original image with background subtracted. This is the image used for calculations in the CSV
* `Background.tif`
  * Stack corresponding to the background removed from the original image
* `flows.tif`
  * optical flow/orientation/cell pose image
* `masks.tif`
  * Image file where pixel intensity corresponds to the cell identity
* `____ nuclei/cyto.svg`
  * Plot showing original image, segmentation outlines, and optical flows
* `____ Cell Intensity Values.csv`
  * Calculated image values after segmentation

## Cell Intensity CSV File
A CSV file is outputted at the end of the segmentation, containing the following columns.
* `Mask Number`
  * Identifies the cell mask. Ordered from highest Y-value pixel.
* `Center X`
  * Center pixel X-value of flows
* `Center Y`
  * Center pixel Y-value of flows
* `Area (px^2)` or `Area (px^3)`
  * number of pixels with mask name
* `CHANNEL_NAME (Magnitude/px^2)` or `CHANNEL_NAME (Magnitude/px^3)`
  * Sum of pixel values within mask area on `Background Subtracted Image.tif` / `Area`
* `CHANNEL_NAME Standard Deviation (Magnitude/px^2)` or `CHANNEL_NAME Standard Deviation (Magnitude/px^3)`
  * Square root of variance of pixel values within mask area on `Background Subtracted Image.tif` / `Area`
* `CHANNEL_NAME Background Intensity (Magnitude/px^2)` or `CHANNEL_NAME Background Intensity (Magnitude/px^3)`
  * Sum of pixel values within mask area on `Background.tif` / `Area`
