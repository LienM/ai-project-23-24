# Installation

The inside the [requirments](./requirements.txt) file work for me. Depending on the GPU you have and what CUDA versions it
supports you might have to install different versions of dgl and pytorch. 

Version selector for (dgl)[https://www.dgl.ai/pages/start.html ] and for (pytorch)[https://pytorch.org/get-started/locally/ ]

At some point I needed an older version of pytorch to get the correct CUDA versions, those you can find (here)[https://pytorch.org/get-started/previous-versions/ ]

# Getting it the DGSR examples running
Change the shebang in the .sh files in DGSR so it points to the correct bash environment

Set value of --k_hop to 3 to match the loading of the dataset