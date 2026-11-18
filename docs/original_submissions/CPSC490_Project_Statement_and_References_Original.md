 Description: 213-csuf-logo-horiz-4c

**CPSC 490 - Undergraduate Seminar in Computer Science**
**Project Statement & References Assignment**
**PLEASE UPLOAD YOUR REPORT TO CANVAS**
**NO PAPER REPORTS**
**Every Individual needs to submit**

**Team Member Names: Tristan Caluag, Billy Leong, Kyle Ly, Eric Tran**







**Grading Criteria:**

| **Section**        | **Earned Points** | **Possible Points** |
| ------------------ | ----------------- | ------------------- |
| **Project Idea 1** | <br>              | 20                  |
| **Project Idea 2** | <br>              | 20                  |
| **Project Idea 3** | <br>              | 20                  |
| **Project Idea 4** | <br>              | 20                  |
| **Project Idea 5** | <br>              | 20                  |
| **Total:**         | <br>              | 100                 |



**Objective**
The main objective is to develop a project statement for each idea. Additionally, this assignment aims to identify whether your project has been done before and/or if it is possible to do it in one semester. If the project idea has been done before, this research will help you identify how your team could improve upon this idea. 

**Instructions**
In this assignment, you will create a project statement (see Lecture 2) for each of your 5 project ideas. 
For each section below, create a project statement and cite the sources (use IEEE Format). You only need 2 or 3 references for each project idea. Feel free to split the work and write student name amongst your team members but combine the work into one document every individual needs to submit.



Project Statement for Project 1 (writer name):

Statement of Need (writer name):

Preliminary Requirements (writer name):

Basic Limitations (writer name):

References (writer name):





---

# Project Idea 1 (Tracy Tran)

Project Statement for Project 1 (writer name): Tracy Tran
This project proposes an AI-powered workout form analysis application that provides personalized feedback based on an individual’s body proportions and body structure. Using a camera, the app will record a user performing an exercise and use pose estimation to track body position, joint angles, and movement [3]. Similar AI-based exercise correction systems have been developed, but this project aims to improve upon them by considering an individual’s body proportions when evaluating form [3]. Unlike traditional fitness videos or social media reels that may demonstrate one standard version of an exercise, the app will consider differences such as torso, femur, and leg proportions. The system will use these measurements to determine whether the user’s movement is appropriate for their individual body structure. It will then identify potential form issues and provide suggestions for improvement. The goal is to make personalized workout guidance more accessible to people who may not have access to or cannot afford a personal trainer.
Statement of Need (writer name): Tracy Tran 

Exercise, health, fitness, and bodybuilding have become increasingly popular on social media, with fitness influencers constantly sharing new exercises and techniques. However, influencers demonstrate exercises based on their own bodies and cannot always take into consideration that everyone has different body proportions and anatomy. Differences in height, torso length, femur length, leg length, and mobility can affect how an exercise should naturally be performed for each individual. When someone copies an influencer’s exact form, it may not be appropriate for their individual body structure, potentially leading to improper technique. Improper exercise technique can become especially concerning when performing exercises with heavier weights [1], [2]. Beginners may also have difficulty recognizing whether their form is appropriate for their own body. Therefore, there is a need for a personalized tool that evaluates exercise form by taking an individual’s anatomy and body proportions into consideration, allowing it to provide feedback based on how that person’s body naturally moves rather than comparing everyone to the same standard of “correct” form.
Preliminary Requirements (writer name): Tracy Tran 
The project will require a camera or image sensor to capture the user performing an exercise and a platform that can run on a smartphone or laptop. A user interface will allow users to record exercises, enter personal information, and view their results. An AI-based body and pose tracking system will identify body landmarks, calculate joint angles, and estimate proportions such as torso, femur, and leg length [3]. If the camera cannot accurately determine these proportions, the user interface will allow users to manually enter body-part measurements to personalize the analysis. The system will also require research-based exercise mechanics that define acceptable movement patterns for each supported exercise. The AI will compare the user's movement to these guidelines while taking their individual measurements and body proportions into consideration, then provide personalized feedback about possible form issues and improvements.
Basic Limitations (writer name): Tracy Tran
The accuracy of the application may be limited by camera quality, camera angle, lighting, clothing, or parts of the user’s body being blocked from view [3]. The system may also have difficulty accurately estimating body-part measurements, requiring users to manually enter measurements such as torso, femur, and leg length. The application will mainly consider visible body proportions and will not be able to fully account for factors such as muscle mass, flexibility, joint mobility, previous injuries, disabilities, medical conditions, or internal bone and joint structure. These factors may affect how an individual safely performs an exercise and cannot be accurately determined through a standard camera. The accuracy of the feedback will also depend on the research and exercise guidelines used to define proper form. Due to the time limitations of one semester, the application may initially support only a small number of exercises and should not be considered a replacement for a certified trainer, physical therapist, or medical professional.
References (writer name): Tracy Tran
[1] M. J.-Y. Tung, G. A. Lantz, A. D. Lopes, and L. Berglund, “Injuries in weightlifting and powerlifting: An updated systematic review,” BMJ Open Sport & Exercise Medicine, vol. 10, no. 4, Art. no. e001884, 2024, doi: 10.1136/bmjsem-2023-001884.
[2] K. Golshani, M. E. Cinque, P. O’Halloran, K. Softness, L. Keeling, and J. R. Macdonell, “Upper extremity weightlifting injuries: Diagnosis and management,” Journal of Orthopaedics, vol. 15, no. 1, pp. 24–27, 2018, doi: 10.1016/j.jor.2017.11.005.
[3] H. Jubair, R. T. Chowdhury, F. Akter, I. S. Sami, N. Nawal, S. T. Arshi, and P. Shil, “Machine Learning for Real-Time Exercise Correction and Injury Prevention: A Systematic Review,” Authorea Preprints, 2025, doi: 10.22541/au.174466167.73334339/v1.





---

# Project Idea 2 (Kyle Ly)

Project Statement for Project 2 (writer name): Kyle Ly
The purpose of this project is to use machine learning and a camera to analyze a user's physical features, including body proportions, hair color, and skin tone, and to recommend the styles that best suit them. The goal is to help people who have difficulty finding a look that fits them. The system's output would include hairstyle suggestions based on face shape and color palettes that complement the user's skin tone. 
Statement of Need (writer name): Kyle Ly 
Social media plays a large role in daily life, and many people feel pressure to stand out or to follow the styles of their favorite influencers. Copying an influencer's style often does not translate well, however, because a look that suits one person may not suit another. This leaves many people unsure of what actually fits them or how to present themselves at their best. Confidence in one's appearance matters, since people who feel good about how they look often feel more comfortable and perform better in their daily tasks. 
Preliminary Requirements (writer name): Kyle Ly 
At a broad level, the project would require a camera or similar image sensor, a software platform that runs on common devices such as phones and laptops, a user interface, facial-tracking and image-processing libraries, and a dataset that maps user features to suitable styles. 
Basic Limitations (writer name): Kyle Ly 
Several factors could constrain the system's accuracy. Image quality, camera angle, and lighting conditions (including changes caused by weather or indoor versus outdoor settings) would all affect how well the system reads a user's features. Results may also vary by region, since style preferences and norms differ across cultures and locations. A significant concern is algorithmic bias: because the model relies on training data, it could produce skewed or unfair recommendations if that data does not represent a wide range of users. 
References (writer name): Kyle Ly 
[1] [author(s)], "DeepColor: A CNN-based skin tone detection and personalized color recommendation system," in *Proc. 2025 3rd DMIHER Int. Conf. Artif. Intell. Healthcare, Educ. Ind. (IDICAIHEI)*, Wardha, India, 2025, doi: 10.1109/IDICAIHEI65991.2025.11377529.
[2] Google, "Face detection guide," Google AI Edge, Google for Developers. Accessed: Aug. 27, 2026. [Online]. Available:[ https://developers.google.com/edge/mediapipe/solutions/vision/face\_detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)

---

# Project Idea 3 (Tristan Caluag)

## Project Statement

The goal for this proposal is to implement a smart home system that can be controlled using hand gestures utilizing real-time computer vision in order to recognize those gestures as well as translate them into commands controlling any registered household devices such as lights, fans, or other appliances.

## Commercial/Industry Need

The main need for this would be to provide more accessibility for people with any disabilities or impairments that limit their motor control and struggle with small switches, remotes, or touchscreens. This also fits the needs of people who are nonverbal or have speech disabilities and cannot use smart home controls that are mainly voice-controlled. Voice may not work well in a loud environment, and hands may be dirty or contaminated, unable to touch the screen. Currently, there are not many smart home systems, such as Alexa and Google Home, that implement this feature and are primarily voice or touch-controlled, in which a project like this could fill that gap.

## Project Requirements

This project would require a type of sensor or camera to capture physical movements and recognize gestures. Machine learning algorithms would be used to analyze captured movement data, identify patterns, and classify gestures using libraries with predefined gesture sets. Processors capable of handling the computational load of real-time gesture recognition are also needed. Implementing a user interface that allows the user to interact with and modify the system to their needs may also be a requirement [1]. The additional hardware that could be required includes a processing unit and modules to connect smart devices for the system to interact with.

## Expected Limitations

Limitations could include both the environment as well as technical constraints. If lighting is low or uneven, it could be hard to detect any hand gestures. The camera’s field of view or angle could also affect hand and gesture detection. Anything in the background could also affect detection reliability depending on the model and training data used. Gesture vocabulary can be limited as there are only a handful of distinct gestures. Multiple people within view could also be hard to distinguish and can cause conflicting or accidental calls to commands. Hand movements made every day could also cause false positives or negatives in detection. Overlapping, occlusion, and interpretation challenges are common in gesture recognition technology but can be solved depending on the data sets used to train the system [2].

## References

[1]
“Gesture Recognition Software,” *Meegle.com*, Feb. 7, 2026. [https://www.meegle.com/en\_us/topics/computer-vision/gesture-recognition-software](https://www.meegle.com/en_us/topics/computer-vision/gesture-recognition-software) (accessed Aug. 26, 2026).
[2] “Hand Tracking and Gesture Recognition Using AI: Applications and Challenges,” *Intellias*, Aug. 14, 2020. 
[https://intellias.com/hand-tracking-and-gesture-recognition-using-ai-applications-and-challenges/](https://intellias.com/hand-tracking-and-gesture-recognition-using-ai-applications-and-challenges/) (accessed Aug. 26, 2026).

---



Project Idea 4 (Billy Leong)

## Project Statement

This proposal is centered around Unmanned Aerial Vehicle (UAV) drone swarm technology with the addition of dynamic follow-me capability. Individual UAVs should be able to dynamically and autonomously place themselves around a tracking subject while in motion in positions relative to each other and the subject itself. Commands, either directed by the tracking subject or by an external operator, may change the swarm’s actions or relative formation at will.

## Commercial/Industry Need

While follow-me and drone swarm technology both already exist, both technologies have yet to be combined at a commercial level. Dynamic drone swarm technology is currently only achieved in experimental trials. If chosen for further development, this proposal plans to bring the multiple technologies together into a singular package which can be used as a basis for further technologies which rely on tight multi-drone operations while limiting the number of operators necessary for coordination or operation.

For example film crews who may wish for multiple aerial shots of a certain scene may use this technology where typical drone-based capture is infeasible because of complications or safety concerns with multiple drones.

## Preliminary Requirements

This project will require at least three programmable drones that are radar, Global Positioning System (GPS), bluetooth, and camera equipped, with additional modifications if deemed necessary. A UAV permit and registration for the multiple drones will be required under federal and state regulations. Artificial Intelligence (AI) models for both follow-me and multi-drone capabilities are both publically available and can be either acquired or developed for this project. To execute the AI models, a portable computer which will serve as the central director for the system is also required. An adequately large non-public staging area to test the system will need to be found.

## Expected Limitations

Funding and capital is currently believed to be the largest limitation of this proposal; preliminary estimates place the probable cost of acquiring or constructing three prototype drones with the necessary capabilities for the project between 3,000 to 5,000 dollars [1]. Additionally a portable computer, likely a laptop with the necessary hardware to run the system is expected to cost between 1,000 to 2,000 dollars. Assuming the need for a budgetary buffer of around 2,000 dollars, total research and development costs for this proposal will fall somewhere between 6,000 - 9,000 dollars at the minimum, not including additional equipment or expenses.

Safety concerns over the operation of multiple drones in close proximity to a human subject is significant. In addition to flight limitations according to Federal Aviation Administration and California state standards [2], GPS receivers are known to have an accuracy of within a 5 meter radius which greatly increases the chances of collision. It is expected that testing can only be safely conducted in a closed environment. 

## References

[1] “PiHawk Landing Page” Drone Dojo. Accessed: Aug. 16, 2026 [Online]. Available: https\://dojofordrones.com/raspberry-pihawk-drone-kit/ 
[2] “California Drone Laws 2026: Rules, Permits & Where to Fly.” Fly USI (Unmanned Safety Institute). Accessed: Aug. 16, 2026 [Online]. Available: https\://www\.flyusi.org/guides/drone-laws/california

---

# Project Idea 5 (Eric Tran)

**Project Statement (Eric Tran)**
This project uses machine learning to design a smart home system to identify and record the faces of people attempting to enter a property using facial recognition. If the system detects an unrecognized or unauthorized individual, it can automatically lock the gate or door, preventing the person from gaining access. The ultimate goal for this smart home system is to improve home security by saving the faces of each individual into the system and differentiating their priority.
**Statement of Need (Eric Tran) **
There have been cases of homes being broken into by intruders, and in many of these situations, residents are caught off guard and unprepared. The threat of home intrusions can cause residents to feel anxious and distrustful of the people around them. While homeowners can install security cameras to monitor their property and identify potential intruders, cameras alone do not necessarily prevent a break-in from occurring. By implementing this project, homeowners are provided with an additional layer of security by combining facial recognition, automated access control, and real-time threat detection.
**Preliminary Requirements (Eric Tran)**
This type of design would require a gate/door equipped with an automatic lock mechanism that can be configured according to the homeowner’s preferred security setting. The system should allow the homeowner to register the faces that come in contact with the sensor and customize the settings for those who are permitted to access beyond the security system. That will then be linked to a CCTV system to capture human activity within its radius, followed by a smart app to notify the user about it. Machine learning can help sort out and detect any suspicious activity that differs from the homeowner’s usual behavior.
**Basic Limitations (Eric Tran) **
Several limitations while using this design include camera distortion during nighttime, awkward camera angles, and potentially confusing an authorized individual due to low resolution or having similar physical features to another person. These are the most recognizable factors that can interfere with the use of facial recognition, complicating the permissions between unauthorized or permitted individuals. 
**References (Eric Tran) **
[1] “Raspberry Pi-Based Face Recognition Door Lock System,” MDPI, May 20, 2025
[https://www.mdpi.com/2624-831X/6/2/31](https://www.mdpi.com/2624-831X/6/2/31) (Accessed: Aug 29, 2026)
[2] “The DEVELOPMENT OF IOT BASED SMART DOOR LOCK AND FIRE ALERT SYSTEM USING FACIAL RECOGNITION,” JACTA, July 30, 2025
[https://jacta.utem.edu.my/jacta/article/view/5296](https://jacta.utem.edu.my/jacta/article/view/5296) (Accessed: Aug 29, 2026)