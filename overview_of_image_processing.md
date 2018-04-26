
## CBIS Images

The CBIS images were saved as Dicom images, which is a format used to save medical images. To read them I downloaded a free Dicom viewer, MicroDicom, from http://www.microdicom.com. I used MicroDicom to export the images as JPEGs. I attempted to export the images to a standard size so they would all be the same, but decided to export each image at it's own size and then extract the ROIs.

MicroDicom exports each image to it's own directory where the patient and scan information are in the directory names, so I wrote a function, rename_and_copy_files in mammo_utils, which traverses the directory structure and copies each image to a single directory with the image and patient information in the image name. The JPEGs of each image were put in one directory and the masks in another. Some images have multiple ROIs, which means multiple masks.

I had considered using the pre-cropped and zoomed ROI images provided with the CBIS-DDSM dataset, but the images were all different sizes and aspect ratios. I wanted the training data to be as standardized as possible, so I decided to crop the ROIs out of the images myself so I would have more control.

The ROIs were extracted from the images using multiple methods to create multiple datasets. The techniques used and the datasets created are detailed in the ReadMe.md file and the code used to do this are in the crop_cbis_images_x.ipynb notebooks.
 
The CBIS-DDSM data was already divided into training and test data. At first I combined all the images from all categories, shuffled it and then divided it into training and test data. I then realized that having different versions of the same images potentially included in both the training and test sets could defeat the purpose of having separate datasets, so I decided to keep the data divided as it was originally. This had the unfortunate side effect of limiting the size of the test and validation data sets while increasing the size of the training dataset.

In order to avoid overlap between the training and validation data the test images were divided in half, in alphabetical order before being shuffled. This would allow at most 1 image to appear in both the test and validation datasets, and that image would be randomly cropped, flipped and rotated so I thought that was acceptable.

The CBIS-DDSM images were combined with the normal DDSM images. The training data was written to tfrecords files while the test and validation data were saved as npy files.

## DDSM Images

The DDSM dataset, from which the normal images were taken, is saved as Lossless JPEGS, a format which has not been maintained since the 1990s. Decoding these images to PNGs was a long and complicated processs, which was greatly sped up by the use of the following tools:

 - Decompressing for LJPEG Images - https://github.com/zizo202/Decompressing-For-LJPEG-image
 - jpeg.exe - an executable specifically for processing DDSM images, written by Chris Rose

To run this on my Windows laptop, I had to install Cygwin, configure the cygwin environment, download several exe files and then run the Python script from the GitHub repo above.

The conversion script first uncompresses the LJPEG files to an LJPEG.1 file, which is then converted to a raw PNM file. This step also takes into account the specific type of scanner used to create the images, as the images need to be normalized differently depending on the scanner. Finally the PNM files are converted to PNGs.

The DDSM data came with each scan in separate subfolders. To streamline the process, I used a script to copy the contents of each subdirectory to a single directory, which I could then run through the conversion process. Once the images were converted I copied the PNGs to a separate directory.

One of the scanners (DBA) had what I presume to be personal information on the scans cut out to leave chunks of white pixels. To remove the possibility of this being used by a ConvNet I programmatically replaced all white pixels with black pixels after I had reviewed the images to ensure that no (or almost no) pure white pixels would appear.

Once the images had been converted and saved into single directories, function mammo_utils.create_slices() was used to create usable data. This function takes the path to the directory as input, generates a list of the images in the directory, and then feeds each image to mammo_utils.slice_normal_image(). This function processes each image as follows:

 1. The PNG image is read in and converted to RGB. The image is then scaled down to half size.
 2. A 7% margin is trimmed from the sides of each image to eliminate the white borders that occur in many scans.
 3. Each image is cut into 299x299 tiles with a stride of 299.
 4. The list of tiles is looped through and if the tile meets certain conditions it is added to the list of usable tiles.
 5. The conditions each tile must meet are to ensure that the image does not contain mostly black background and contains usable content. This is done by setting lower and upper thresholds on the mean and variance of each image. The thresholds were determined by manually reviewing the tiles and their corresponding means and variances.
 
Once the tiles are created, the tiles from each of the scanners are combined, shuffled and saved in multiple batches to keep the files at reasonable sizes.

## MIAS Data

This project was intended to work with the MIAS data, but the dataset proved far too small to use as training data. 

The MIAS data consists of 1024x1024 images, with the scan horizontally centered in the image. The DDSM images were full sized. In order to have the images on similar scales I calculated the average size of the DDSM images, factored in the fact that I would be scaling each image down by half the get the ROI to fit in a 299x299 tile, and came up with a factor of 2.58 by which the MIAS images need to be scaled up.

The MIAS data had each ROI identified by a center point and a radius whereas the CBIS-DDSM data had a mask to highlight the ROI. For the sake of reusability the function to extract the masks from the CBIS-DDSM mask images also returned a center point with a height and a width.

In order to keep the MIAS data as similar to the DDSM data as possible I attempted to use the same methods to process the MIAS images. The scripts to create the data from the images was as similar to the DDSM scripts as possible.

I had planned to either include the MIAS data in the training dataset or to use it as a test set, but a visual review of the data seemed to indicate that it is somehow different than the DDSM data. I can't put my finger on exactly how, but it seems to me that the normal MIAS images do not resemble the normal DDSM images.

I still used the MIAS data as a test data set, where my suspicion seems to be confirmed as models which perform well on the DDSM data have terrible accuracy on the MIAS data. The recall on the MIAS data is excellent, but the precision is terrible.
