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
        end_frame=121,
        Hough_threshold=30,
        Hough_gap=10,
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
    hglinev = video.copy()

    tiff_loader = TiffLoader(video)
    if start_frame not in frame2line:
        print("Please draw a line to specify a microtubule")
        return []
    # select a line to detect the corresponding microtubule
    for i in range(start_frame, min(end_frame, len(video))):
        img = tiff_loader.tiff_gray_image[i]
        if i in frame2line:
            line = frame2line[i]
        end_points, skltn, thres_img, denoise, temp, temp1, first_bin, hglines, hgline = image_processing.detectLine(img, line,
                                                                                                             struct_size, Hough_gap, thres_ratio, Hough_threshold)
        if end_points == 'err':
            print("error occured")
            break
        line = [[i, end_points[0][0], end_points[0][1]], [i, end_points[1][0], end_points[1][1]]]
        video[i] = thres_img * 257
        tempv[i] = temp * 257
        tempv1[i] = temp1 * 257
        binv[i] = first_bin * 257
        # thinv[i] = skltn * 257
        hglinesv[i] = hglines * 257
        hglinev[i] = hgline * 257
        print(i)
        print('new end points = ', end_points)

    layer_type = 'image'
    metadata = {
        'name': 'temp1',
        'colormap': 'gray'
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
        'name': 'segment',
        'colormap': 'red',
        'blending': 'additive'
    }
    metadata4 = {
        'name': 'ghlines',
        'colormap': 'gray'
    }
    metadata5 = {
        'name': 'tgt_ghline',
        'colormap': 'gray'
    }

    return [(tempv1, metadata, layer_type), (tempv, metadata1, layer_type), (binv, metadata2, layer_type), (video, metadata3, layer_type),
            (hglinesv, metadata4, layer_type), (hglinev, metadata5, layer_type)]


@magicgui(
    call_button="Darken"
)
def darken_img(
        dark_layer: 'napari.layers.Shapes',
        image_layer: 'napari.layers.Image',
        dark_ratio=84,
        start_frame=0,
        end_frame=68
) -> napari.types.LayerDataTuple:
    # the whole output img may look not like the original one. It is the Napari's displaying issue but
    # the pixel values are the same.

    # prepare video data: np.array
    video = np.array(image_layer.data)
    tiff_loader = TiffLoader(video)
    polygon = dark_layer.data[0]
    polygon = polygon[:, 1:]
    for i in range(start_frame, min(end_frame, len(video))):
        img = tiff_loader.tiff_gray_image[i]
        darken(polygon, img, dark_ratio)
        video[i] = img * 257
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
viewer.window.add_dock_widget(darken_img)
viewer.window.add_dock_widget(save)
napari.run()
