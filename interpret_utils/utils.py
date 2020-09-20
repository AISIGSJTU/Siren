# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import os
# import PIL.Image
# import shut


###############################################################################
# Download utilities
###############################################################################


def download(url, filename):
    if not os.path.exists(filename):
        print("Download: %s ---> %s" % (url, filename))
        response = six.moves.urllib.request.urlopen(url)
        with open(filename, 'wb') as out_file:
        	shutil.copyfileobj(response, out_file)

###############################################################################
# Plot utility
###############################################################################


def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    return ret


def get_imagenet_data(size=224):
    base_dir = os.path.dirname(__file__)
    # ImageNet 2012 validation set images?
    with open(os.path.join(base_dir, "images", "ground_truth_val2012")) as f:
        ground_truth_val2012 = {x.split()[0]: int(x.split()[1])
                                for x in f.readlines() if len(x.strip()) > 0} 
    with open(os.path.join(base_dir, "images", "synset_id_to_class")) as f:
        synset_to_class = {x.split()[1]: int(x.split()[0])
                           for x in f.readlines() if len(x.strip()) > 0} 
    with open(os.path.join(base_dir, "images", "imagenet_label_mapping")) as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}

    def get_class(f):
        # File from ImageNet 2012 validation set
        ret = ground_truth_val2012.get(f, None)
        if ret is None:
            # File from ImageNet training sets
            ret = synset_to_class.get(f.split("_")[0], None)
        if ret is None:
            # Random JPEG file
            ret = "--"
        return ret

    images = [(load_image(os.path.join(base_dir, "images", f), size),
               get_class(f))
              for f in os.listdir(os.path.join(base_dir, "images"))
              if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")]
    return images, image_label_mapping


def plot_image_grid(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    dpi=224):
    n_rows = len(grid)
    n_cols = len(grid[0])

    plt.clf()
    plt.rc("font", family="sans-serif")

    plt.figure(figsize = (n_cols+1, n_rows+1)) #TODO figsize
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows+1, n_cols], loc=[r+1,c])
            ax.imshow(grid[r][c], interpolation='none') #TODO. controlled color mapping wrt all grid entries, or individually. make input param
            ax.set_xticks([])
            ax.set_yticks([])

            if not r: #column labels
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom',
                                 fontsize=10
                                 )
                            
                    # print('skip')

            if not c: #row labels
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                ax.set_ylabel(''.join(txt_left),
                              rotation=0,
                              verticalalignment='center',
                              horizontalalignment='right',
                              )
            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(''.join(txt_right),
                                  rotation=0,
                                  verticalalignment='center',
                                  horizontalalignment='left',
                                  fontsize=10
                                   )

    if file_name is None:
        plt.show()
    else:
        print ('saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi)
