# TODO:

## Step 1
- strip all ETL data transformations from start of custom monitors (so they function like OOTB monitors with some added BMS CHIP dimensional data)
- For reference, I have included the OOTB monitors we wish our custom monitors to resemble (with most of the script being the #modelop.init and #modelop.metrics functions. Take a look at the OOTB_monitors in the modelop\ootb_monitors to see how the custom monitors should be built (with a .py file, .dmn file, metadata.json file (optional), README.md file, and a required_assets.json)

## Step 2:
- place etl transformation code to produce master dataset in separate CHIP_mtr_data monitor repository. This will be used to generate the master dataset and the baceline and comparator datasets for all 3 monitors. We can produce csv versions for all datasets in this repository, and don't need to create json versions
- The data assets need to live in the implementation assets
- We can repurpose the CHIP_data subdir to be to where we position all assets for the CHIP_mtr_data

## Step 3:
- Make a required_assets file in the monitor repo so the monitor requires an input asset. Review ModelOp_Center_Custom_Monitor_Developer_Intro_Training_Jan-2024.pptx.pdf to ensure we are including the correct file assets in each of the custom monitor subdirs
