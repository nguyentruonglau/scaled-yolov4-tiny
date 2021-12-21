import sys,os; sys.path.append('..')

from synthetic.draw_utils import recursive_parse_xml_to_dict
from synthetic.gen_utils import ensure_dir
from imutils.paths import list_files
from preprocess import kmeans, avg_iou

from lxml import etree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-xml-dir', type=str, default='../database/dataset/Car/Annotations', help='directory contain xml annotation files')
    parser.add_argument('--output_dir', type=str, default='../database/logdata')
    parser.add_argument('--input-shape', type=tuple, default=(512, 512), help='input shape of model')
    parser.add_argument('--n-clusters', type=int, default=6, help='Number of clusters for kmean anchor, 6 with akaDET-S, 12 with akaDET-B')
    return parser.parse_args()


def get_data_infor(input_annotation_path):
    #get infor xml file
    with open(input_annotation_path, 'rb') as file:
        xml_str = file.read()
    xml = etree.fromstring(xml_str)
    infors = recursive_parse_xml_to_dict(xml)['annotation']

    width = int(infors['size']['width'])
    height = int(infors['size']['height'])

    filename = infors['filename']
    #get bbox for drawing
    bboxs = []; objectnames = []
    if 'object' in infors:
        for obj in infors['object']:
            bboxs.append([
                float(obj['bndbox']['xmin']),
                float(obj['bndbox']['ymin']),
                float(obj['bndbox']['xmax']),
                float(obj['bndbox']['ymax'])
                ]
            )
            objectnames.append(obj['name'])
    return filename, bboxs, objectnames, width, height


def crawl_data2df(args):
    """Organize crawled data into dataframe
    """
    print('[INFOR]: Start crawling...')

    annotation_paths = list(list_files(args.input_xml_dir))
    if not annotation_paths: raise ValueError('[ERROR]: XML files in {} not found'.format(args.input_xml_dir))
    save_data = dict(); object_names = []

    #initialization
    save_data['filename'] = []
    save_data['xmin'] = []
    save_data['ymin'] = []
    save_data['xmax'] = []
    save_data['ymax'] = []
    save_data['w_box'] = []
    save_data['h_box'] = []
    save_data['objectname'] = []
    save_data['w_img'] = []
    save_data['h_img'] = []

    for i in tqdm(range(len(annotation_paths))):
        filename, bboxs, objectnames, width, height = get_data_infor(annotation_paths[i])
        bboxs = np.array(bboxs, dtype=np.float64)
        object_names += np.unique(objectnames).tolist()

        #convert bounding box coordinates
        bboxs = resize_boxes(bboxs, src_size=(width, height), dst_size=args.input_shape)
        n = len(objectnames)

        save_data['filename'] += ([filename] + [' ']*(n-1))
        save_data['xmin'] += list(bboxs[:,0])
        save_data['ymin'] += list(bboxs[:,1])
        save_data['xmax'] += list(bboxs[:,2])
        save_data['ymax'] += list(bboxs[:,3])
        save_data['w_box'] += list(np.abs(bboxs[:,2] - bboxs[:,0]))
        save_data['h_box'] += list(np.abs(bboxs[:,3] - bboxs[:,1]))
        save_data['objectname'] += objectnames
        save_data['w_img'] += ([str(width)] + [' ']*(n-1))
        save_data['h_img'] += ([str(height)] + [' ']*(n-1))

    df = pd.DataFrame(save_data)
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join(args.output_dir, 'anno_infor.xlsx'), engine='xlsxwriter', mode='w')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='data', index=False)
    writer.save()

    print('[INFOR]: Saved data to {}'.format(args.output_dir))
    return df, np.unique(object_names)


def draw_image_ratio(df, num_bin):
    """Draw and save aspect ratio of images
    """
    wid_ls = list(df['w_img']); hig_ls = list(df['h_img'])
    while ' ' in wid_ls: wid_ls.remove(' ')
    while ' ' in hig_ls: hig_ls.remove(' ')

    wid_ls = np.array(wid_ls, dtype=float)
    hig_ls = np.array(hig_ls, dtype=float)

    img_aspect_ratio = np.array(wid_ls/hig_ls)
    plt.hist(x=img_aspect_ratio, bins=num_bin, color='#d254d6')
    plt.title('Aspect Ratio Of Images')
    plt.xlabel('aspect ratio')
    plt.ylabel('count')
    # plt.show()
    plt.savefig(os.path.join(args.output_dir, 'img_aspect_ratio.png'))
    plt.close()


def draw_box_ratio(df, num_bin):
    """Draw and save aspect ratio of boxes
    """
    box_aspect_ratio = np.abs(df['xmax'] - df['xmin'])/np.abs(df['ymax'] - df['ymin'])
    plt.hist(x=box_aspect_ratio, bins=num_bin, color='#49c5de')
    plt.title('Aspect Ratio Of Boxes')
    plt.xlabel('aspect ratio')
    plt.ylabel('count')
    # plt.show()
    plt.savefig(os.path.join(args.output_dir, 'box_aspect_ratio.png'))
    plt.close()


def draw_box_count(df, num_bin):
    """Count the number of boxes of each category
    """
    box_arr = np.array(df['objectname'])
    plt.hist(x=box_arr, bins=num_bin, color='#55eda9')
    plt.title('Box Count')
    plt.xlabel('category'); plt.xticks(rotation='vertical')
    plt.ylabel('count')
    # plt.show()
    plt.savefig(os.path.join(args.output_dir, 'box_count.png'))
    plt.close()


def draw_avg_box_area(df):
    """Calculate the averacge area of the boxes for each category
    """
    # Initialization
    dic_name2value = dict()
    uname_arr = np.array(df['objectname'].unique(), dtype=str)
    for name in uname_arr:
        dic_name2value[name] = []

    # Addition
    name_arr = np.array(df['objectname'], dtype=str)
    x_box = np.array(df['w_box'], dtype=float)
    y_box = np.array(df['h_box'], dtype=float)

    for x,y,name in zip(x_box, y_box, name_arr):
        dic_name2value[name].append(x*y)

    value_arr = [np.mean(dic_name2value[name]) for name in uname_arr]

    plt.bar(uname_arr, value_arr, color='#ede661')
    
    plt.title('Average Box Area')
    plt.xlabel('category'); plt.xticks(rotation='vertical')
    plt.ylabel('count')
    # plt.show()
    plt.savefig(os.path.join(args.output_dir, 'avg_box_area.png'))
    plt.close()


def resize_boxes(boxes, src_size, dst_size):
    """Resize boxes by scale and pad_size when resizing images
    """
    scale = np.minimum(dst_size[0]/src_size[0], dst_size[1]/src_size[1])
    src_size = (src_size[0]*scale, src_size[1]*scale)
    pad_size = np.array(dst_size, dtype=int) - np.array(src_size, dtype=int)
    
    boxes = np.array(boxes, dtype=float)
    boxes[:, 0:4] *= scale
    half_pad = pad_size // 2
    boxes[:, 0:4] += np.tile(half_pad,2)
    return boxes

def get_anchors(df, uni_names):
    """Get anchor boxes for training phase
    """
    w_box = np.array(df['w_box'])/args.input_shape[0]
    h_box = np.array(df['h_box'])/args.input_shape[1]

    data = np.concatenate((w_box.reshape(-1,1), h_box.reshape(-1,1)), axis=-1)
    anchor_box = kmeans(data, k=args.n_clusters)

    with open(os.path.join(args.output_dir, 'anchor_box.txt'), 'w') as f:

        f.writelines("Accuracy: {:.2f}% \n\n".format(avg_iou(data, anchor_box) * 100))

        f.writelines("Boxes: \n")
        #write bounding boxes
        for box in anchor_box:
            f.writelines(str((np.array(box)*args.input_shape[0]).astype(int))+'\n')

        #write labels
        f.writelines("\nLabel: \n")
        for label in uni_names:
        	f.writelines(str(label)+'\n')


def main(args):
    # Crawl data and organize crawled data into dataframe
    df, uni_names = crawl_data2df(args); num_bin = len(uni_names)
    print('[INFOR]: Doing data statistics...')
    # Draw, save aspect ratio of images and boxes
    draw_image_ratio(df, num_bin); draw_box_ratio(df, num_bin)
    # Count the number of boxes of each category
    draw_box_count(df, num_bin)
    # Calculate the averacge area of the boxes for each category and get anchor box
    draw_avg_box_area(df); get_anchors(df, uni_names)
    print('[INFOR]: Completion!')


if __name__ == '__main__':
    args = parser()
    print("\ninput-xml_dir = ", args.input_xml_dir)
    print("output-dir = ", args.output_dir)
    print("n-clusters = ", args.n_clusters)
    print("input-shape = ", args.input_shape, '\n')
    main(args)
