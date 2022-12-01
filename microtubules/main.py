import PIL.ImageDraw
import napari
import numpy as np
from lib.load_data import TiffLoader
from lib import image_processing
from magicgui import magicgui
from lib.img_helper import darken
import typing


@magicgui(
    call_button="Calculate",
    options={"choices": ['show segmentation', 'show skeleton', 'test']}
)
def main(
        draw_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        struct_size=7,
        start_frame=0,
        end_frame=71,
        Hough_gap=20,
        thres_ratio=1.0,
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
    tempv1 = video.copy()
    binv = video.copy()
    thinv = video.copy()
    hglinesv = video.copy()

    tiff_loader = TiffLoader(video)
    if start_frame not in frame2line:
        print("Please draw a line to specify a microtubule")
        return []
    # select a line to detect the corresponding microtubule
    for i in range(start_frame, end_frame):
        img = tiff_loader.tiff_gray_image[i]
        if i in frame2line:
            line = frame2line[i]
        end_points, skltn, thres_img, denoise, temp, temp1, first_bin, hglines = image_processing.detectLine(img, line,
                                                                                                             struct_size, Hough_gap, thres_ratio)
        if end_points == 'err':
            print("error occured")
            break
        line = [[i, end_points[0][0], end_points[0][1]], [i, end_points[1][0], end_points[1][1]]]
        video[i] = thres_img * 257
        tempv[i] = temp * 257
        binv[i] = first_bin * 257
        thinv[i] = skltn * 257
        hglinesv[i] = hglines * 257
        tempv1[i] = temp1 * 257
        print(i)
        print('new end points = ', end_points)

    layer_type = 'image'
    metadata = {
        'name': 'segment',
        'colormap': 'blue',
        'opacity': 0.4
    }
    metadata1 = {
        'name': 'temp',
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
    metadata4 = {
        'name': 'temp1',
        'colormap': 'gray'
    }

    return [(tempv1, metadata4, layer_type), (tempv, metadata1, layer_type), (binv, metadata2, layer_type), (video, metadata, layer_type),
            (hglinesv, metadata3, layer_type)]



@magicgui(
    call_button="Darken"
)
def darken(
        dark_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        dark_ratio=84,
        start_frame=120,
        end_frame=68
) -> napari.types.LayerDataTuple:
    # prepare video data: np.array
    video = np.array(image_layer.data)
    tiff_loader = TiffLoader(video)
    polygon = dark_layer.data[0]
    # polygon = np.array([[41., 253.92413287, 76.63881398],
    #        [41., 269.68282815, 133.37011699],
    #        [41., 250.77239381, 249.19652729],
    #        [41., 212.16359038, 256.28794016],
    #        [41., 200.34456892, 221.61881055],
    #        [41., 198.76869939, 153.06848609],
    #        [41., 216.89119896, 92.39750926],
    #        [41., 233.437829, 73.48707493]])
    polygon = polygon[:, 1:]
    for i in range(start_frame, min(end_frame, len(video))):
        img = tiff_loader.tiff_gray_image[i]
        dark_img = darken(polygon, img, dark_ratio)
        video[i] = dark_img * 257
        print(i)

    layer_type = 'image'
    metadata = {
        'name': 'darken',
        'colormap': 'gray'
    }

    return video, metadata, layer_type


@magicgui(call_button="Save to local", )
def save(image_layer: 'napari.layers.Image', ):
    image_layer.save('segmentation.tif', plugin='builtins')


viewer = napari.Viewer()
viewer.window.add_dock_widget(main)
viewer.window.add_dock_widget(darken)
viewer.window.add_dock_widget(save)
napari.run()
