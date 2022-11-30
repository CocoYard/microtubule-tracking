import PIL.ImageDraw
import napari
from skimage import io
from PIL import Image
import numpy as np
from lib.load_data import TiffLoader
from lib import image_processing
from magicgui import magic_factory
from magicgui import magicgui
import typing


@magicgui(
    call_button="Calculate",
    options={"choices": ['show segmentation', 'show skeleton', 'test']}
)
def widget_demo(
        draw_layer: 'napari.layers.Shapes',
        dark_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        dark_ratio=84,
        struct_size=10,
        start_frame=41,
        end_frame=42,
        options='show segmentation'
) -> typing.List[napari.types.LayerDataTuple]:
    frame2line = {}
    if draw_layer is not None:
        lines = draw_layer.data
        for line in lines:
            frame2line[line[0][0]] = line

    # prepare video data: np.array
    video = np.array(image_layer.data)
    tempv = video.copy()
    binv = video.copy()
    thinv = video.copy()
    hglinesv = video.copy()

    tiff_loader = TiffLoader(video)
    polygon = dark_layer.data[0]
    # polygon = np.array([[ 43.        , 253.74613479,  91.11234075],
    #     [ 43.        , 247.48144343, 167.54157533],
    #     [ 43.        , 235.78735289, 183.82977286],
    #     [ 43.        , 234.95206071, 193.85327904],
    #     [ 43.        , 237.87558335, 205.54736957],
    #     [ 43.        , 225.76384672, 241.88257946],
    #     [ 43.        , 220.33444754, 222.67085929],
    #     [ 43.        , 220.75209363, 165.87099097],
    #     [ 43.        , 229.52266153, 119.09462882],
    #     [ 43.        , 239.9638138 ,  96.95938602],
    #     [ 43.        , 247.48144343,  91.11234075]])
    # polygon = np.array([[41., 253.92413287, 76.63881398],
    #        [41., 269.68282815, 133.37011699],
    #        [41., 250.77239381, 249.19652729],
    #        [41., 212.16359038, 256.28794016],
    #        [41., 200.34456892, 221.61881055],
    #        [41., 198.76869939, 153.06848609],
    #        [41., 216.89119896, 92.39750926],
    #        [41., 233.437829, 73.48707493]])
    polygon = polygon[:, 1:]
    # select a line to detect the corresponding microtubule
    for i in range(start_frame, end_frame):
        img0 = tiff_loader.tiff_gray_image[i]
        if i in frame2line:
            line = frame2line[i]
        end_points, skltn, thres_img, denoise, temp, first_bin, hglines = image_processing.detectLine(img0, line, polygon, struct_size, dark_ratio)
        if end_points == 'err':
            print("error occured")
            break
        line = [[i, end_points[0][0], end_points[0][1]], [i, end_points[1][0], end_points[1][1]]]
        video[i] = thres_img * 257
        tempv[i] = temp * 257
        binv[i] = first_bin * 257
        thinv[i] = skltn * 257
        hglinesv[i] = hglines * 257
        print(i)
        print('new end points = ', end_points)


    layer_type = 'image'
    metadata = {
        'name': 'segment',
        'colormap': 'blue',
        'opacity': 0.4
    }
    metadata1 = {
        'name': 'segment2',
        'colormap': 'gray'
    }
    metadata2 = {
        'name': 'initial bin img',
        'colormap': 'gray'
    }
    metadata3 = {
        'name': 'thinning img',
        'colormap': 'gray'
    }

    return [(tempv, metadata1, layer_type), (binv, metadata2, layer_type), (video, metadata, layer_type),
            (hglinesv, metadata3, layer_type)]


@magicgui(call_button="Save to local", )
def save(image_layer: 'napari.layers.Image', ):
    image_layer.save('segmentation.tif', plugin='builtins')


viewer = napari.Viewer()
viewer.window.add_dock_widget(widget_demo)
viewer.window.add_dock_widget(save)
napari.run()
