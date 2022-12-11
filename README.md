# Microtubule Tracking
## Application
1. This is a tool to track microtubules, which are major components of the cytoskeleton.
2. Use the Napari built-in button to add shape, choose the line shape and draw it on the microtubule you want to track on the first frame, and click "Calculate".
3. If you want to use the darken function, use the Napari built-in button to add a shape, choose the polygon shape, draw it on the area you want to darken, and then click "Darken". It usually takes about 30s seconds to compute. After it is finished, remember to change the image layer to "darken" before clicking "Calculate".

## How to Run
1. install python 3.8.9
2. initialize virtual environment by run `python3 -m venv venv` in the directory ./microtubules.
3. activate the virtual environment by run
   1. Windows Powershell: `.\venv\Scripts\activate`
   2. MacOS/Linux Terminal: `. venv/bin/activate`
4. The following only works if you successfully activated the virtual environment, otherwise you need to refer to python's official document for alternative ways.
5. install all the dependencies in requirements.txt by run `pip install -r requirements.txt`
6. Run main.py by run `python main.py`

## Demo


https://user-images.githubusercontent.com/98336316/206876735-db05ef55-8522-4465-ba2e-6737c2d04b37.mp4
