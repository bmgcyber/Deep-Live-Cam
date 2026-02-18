@echo off
rem Run Deep-Live-Cam with virtual camera output enabled.
rem The processed (face-swapped) frames are streamed as a virtual webcam
rem that other apps (Zoom, Discord, Teams, OBS) can select.
rem
rem Windows requirement:
rem   OBS Studio must be installed (https://obsproject.com) - it provides
rem   the virtual camera backend that pyvirtualcam uses.
rem   After installing OBS, make sure to start OBS at least once and
rem   enable the Virtual Camera from OBS's toolbar (Start Virtual Camera).
rem
rem Usage:
rem   run-virtualcam.bat                 (CUDA + virtual cam)
rem   run-virtualcam.bat --many-faces    (swap all faces)
rem
C:\Python312\python.exe run.py --execution-provider cuda --virtual-cam %*
