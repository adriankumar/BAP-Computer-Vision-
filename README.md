# BAP-Computer-Vision-
Part of the Biomedical Society at UTS, the Bionic Arm Project (BAP) involved developing a mechanical hand controlled by a computer vision system.

#Notes
Please note that this code was intended to be made on windows application, the Matplot section uses windows backend settings so any errors on different operating system may just be due to compatability.

This code has 4 componenets: 
1. Computer Vision - Uses camera to capture hand
2. Angle calculation - When hand detected, performs bending joints and lateral angle calculations in real time
3. Matplot - When hand detected, plots the landmark data on a coordinate system
4. BLE connection. - Looks for the microcontroller device to connect and send angles to (however this would not be used as this is specifc to the BAP project so you can comment it out or just let it run in the background where it will stop itself)

Each of these 4 components are spread into functions each with their own nested functions and variables to be used, this for easier organisation and scalability as each of these functions run concurrently due to the multiprocessing implementation, meaning they all run independently of each other, hence they use a shared data class/object to control when to perform their dedicated functions.

# How to use:
1. Create a virtual environment and install required libraries from requirements.txt
2. Copy the 'remastered.py' code in virtual environment and run!
3. Press 'g' to stop all processes and stop the program safely
