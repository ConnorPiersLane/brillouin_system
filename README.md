# brillouin_system
## Background

This repository builds upon and extends two original projects by [Amira Eltony](https://github.com/aeltony):

- [microscope-brillouin-daq](https://github.com/aeltony/microscope-brillouin-daq)
- [macro-brillouin-daq](https://github.com/aeltony/macro-brillouin-daq)

It combines and further develops key features from both repositories.

# Required Installations

### For the Allied vision camera (need for Microscope and Human Interface)
#### 1. Download and Install Allied Vision’s Vimba Software

1. **Download:**  
   Visit the [Allied Vision Vimba download page](https://www.alliedvision.com/en/products/software/vimba/) to obtain the latest version of the Vimba software for your operating system.

2. **Install:**  
   Run the installer and follow the on-screen instructions. By default, Vimba is installed in:  
   - **Windows:** `C:\Program Files\Allied Vision\Vimba_6.0`  
   - *(Installation paths may vary on different operating systems.)*

#### 2. Install the Vimba-Python Package

After installing the Vimba software, the `vimbapython` package is available in the following directory:

C:\Program Files\Allied Vision\Vimba_6.0\VimbaPython\Source


**Step A: Copy the Package to a Writable Location**

Due to permission restrictions in the Program Files directory, copy the entire `Source` directory to a location where you can make changes. For example, copy it to:

C:\Users<YourUsername>\VimbaPython\Source

Replace `<YourUsername>` with your actual Windows username.

For example: `C:\Users\Mandelstam\Documents\Connor\VimbaPython\Source`

**Step B: Install the Package**

pip install "vimbapython @ file:///C:/Users/<YourUsername>/vimba-python/Source"

For example: `pip install vimbapython@file:///C:/Users/Mandelstam/Documents/Connor/VimbaPython/Source`
`
### Spinnaker SDK and PySpin for the Flir Camera (microscope only)
Install the Spinnaker PySpin package 
(Attention, pip install PySpin will install pyspin, this has nothing to do with PySpin).
Requirmement (currently: python 3.10).
For example, I downloaded the python 3.10 spinnaker. In my venv, I than had to run:
`pip install "C:\Users\Mandelstam\Downloads\spinnaker_python-4.2.0.83-cp310-cp310-win_amd64\spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl"
`

## 3. Install the remaining Packages
`pip install -r requirements.txt`
