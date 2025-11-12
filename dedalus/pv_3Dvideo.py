import glob
import argparse
from paraview.simple import *

# Parsing stuff
parser = argparse.ArgumentParser(
    description="Create a video from VTR files stores in a given folder",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "folder", help="folder where are stored all the VTR files")
parser.add_argument(
    "--outputFile", "-o", default="video.avi", help="name (with extension !) of the video file")
parser.add_argument(
    "--frameRate", "-r", default=25, type=int, help="frame rate used for the video")
parser.add_argument(
    "--quality", "-q", default=1, choices=["0", "1", "2"], help="video quality (0:low, 2:high)")
parser.add_argument(
    "--noClip", action="store_true", help="don't use the clips in the video (cube only)")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.folder}/*.vtr"))

# create a new 'XML Rectilinear Grid Reader'
data = XMLRectilinearGridReader(registrationName='mouahahaha_[...].vtr', FileName=files)
data.TimeArray = 'None'

animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()

view = GetActiveViewOrCreate('RenderView')
view.ResetCamera(False, 0.9)

display = Show(data, view, 'UniformGridRepresentation')

if args.noClip:
    # Surface for data for data
    display.Representation = 'Surface'
    ColorBy(display, ('POINTS', 'buoyancy'))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(view, True)
else:
    # Outline for data
    display.Representation = 'Outline'
    display.AmbientColor = [0.7019607843137254, 0.7019607843137254, 0.7019607843137254]
    display.DiffuseColor = [0.7019607843137254, 0.7019607843137254, 0.7019607843137254]

    # Vertical clip (diagonal) - Surface
    clip1 = Clip(registrationName='Clip1', Input=data)
    clip1.ClipType.Normal = [0.707, 0.707, 0.0]
    clip1Display = Show(clip1, view, 'UnstructuredGridRepresentation')
    clip1Display.Representation = 'Surface'
    ColorBy(clip1Display, ('POINTS', 'buoyancy'))
    clip1Display.RescaleTransferFunctionToDataRange(True, False)
    clip1Display.SetScalarBarVisibility(view, True)

    # Horizontal clip (middle) - Surface
    clip2 = Clip(registrationName='Clip2', Input=data)
    clip2.ClipType.Normal = [0.0, 0.0, 1.0]
    clip2Display = Show(clip2, view, 'UnstructuredGridRepresentation')
    clip2Display.Representation = 'Surface'
    ColorBy(clip2Display, ('POINTS', 'buoyancy'))
    clip2Display.RescaleTransferFunctionToDataRange(True, False)
    clip2Display.SetScalarBarVisibility(view, True)

# Colorbar stuff
buoyancyTF2D = GetTransferFunction2D('buoyancy')
buoyancyTF2D.ScalarRangeInitialized = 1
buoyancyTF2D.Range = [0.00015009482740424573, 0.9998499751091003, 0.0, 1.0]

buoyancyLUT = GetColorTransferFunction('buoyancy')
buoyancyLUT.TransferFunction2D = buoyancyTF2D
buoyancyLUT.RGBPoints = [0.00015009482740424573, 0.278431372549, 0.278431372549, 0.858823529412, 0.14310717770768677, 0.0, 0.0, 0.360784313725, 0.2850645607076876, 0.0, 1.0, 1.0, 0.4290213434682519, 0.0, 0.501960784314, 0.0, 0.5709787264682527, 1.0, 1.0, 0.0, 0.7139358093485352, 1.0, 0.380392156863, 0.0, 0.8568928922288178, 0.419607843137, 0.0, 0.0, 0.9998499751091003, 0.878431372549, 0.301960784314, 0.301960784314]
buoyancyLUT.ColorSpace = 'RGB'
buoyancyLUT.ScalarRangeInitialized = 1.0
buoyancyLUT.ApplyPreset('Rainbow Desaturated', True)

buoyancyLUTColorBar = GetScalarBar(buoyancyLUT, view)
buoyancyLUTColorBar.Title = 'buoyancy'
buoyancyLUTColorBar.ComponentTitle = ''
buoyancyLUTColorBar.WindowLocation = 'Any Location'
buoyancyLUTColorBar.Position = [0.8758553274682307, 0.028636884306987402]
buoyancyLUTColorBar.ScalarBarLength = 0.32999999999999996

buoyancyPWF = GetOpacityTransferFunction('buoyancy')
buoyancyPWF.Points = [0.00015009482740424573, 0.0, 0.5, 0.0, 0.9998499751091003, 1.0, 0.5, 0.0]
buoyancyPWF.ScalarRangeInitialized = 1

# Time annotation
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='TimeAnnotation', Input=data)
annotateTimeFilter1.Format = 'Time: {time:1.1f}s'
annotateTimeFilter1.Scale = 0.1
annotateTimeFilter1Display = Show(annotateTimeFilter1, view, 'TextSourceRepresentation')
annotateTimeFilter1Display.WindowLocation = 'Any Location'
annotateTimeFilter1Display.Position = [0.46187683284457476, 0.85]

# Active sources
SetActiveSource(data)
if not args.noClip:
    HideInteractiveWidgets(proxy=clip1.ClipType)
    HideInteractiveWidgets(proxy=clip2.ClipType)
HideInteractiveWidgets(proxy=display.SliceFunction)
HideInteractiveWidgets(proxy=display)


# Layout and view
layout1 = GetLayout()
layout1.SetSize(1023, 873)
view.CameraPosition = [2.4986495539150253, 3.0037498810813394, 1.292867266624508]
view.CameraFocalPoint = [0.49218750000000006, 0.49218750000000006, 0.5000000146610546]
view.CameraViewUp = [-0.14739149545982064, -0.1887511037859152, 0.9709010082833968]
view.CameraParallelScale = 0.8569402061940335
view.Update()

# Save Animation
SaveAnimation(
    filename=args.outputFile, 
    viewOrLayout=view, location=16, ImageResolution=[1020, 872],
    FrameRate=args.frameRate,
    FrameWindow=[0, len(files)], 
    # FFMPEG options
    Quality=args.quality)