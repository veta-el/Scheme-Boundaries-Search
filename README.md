# Scheme-Boundaries-Search
Scheme Boundaries Search

The function of finding significant areas in an image, for example, diagrams or schemes, therefore, the boundaries of these areas. One of the uses is to select only important image fragments to send them to vLLM and reduce the number of tokens used.

The algorithm is based on splitting the image into chunks and determining the Mean Gray Value of each chunk, by the threshold value of which the most significant areas are cut off (that is, those with the least white/empty background).
As input, the function takes img_source - the path to the image or an image in binary form, as well as an additional parameter chunk_len - the size in pixels for the side of the chunk, or it is calculated adaptively based on the size of the source image.
The output returns the mask of the image with the selected significant areas and the image itself with the mask applied.

Two constants are used:
* GENERAL_MGV_THRESHOLD - the MGV threshold value, which corrects which images are allowed for analysis (how much white/empty background is allowed).
* MGV_VALUE_PROPORTION - the parameter from 0 to 1 that defines the threshold value for individual chunks.
The values of these constants need to be experimentally adapted depending on the type of images you are working with (how much white background, which details of the image are more important etc).
