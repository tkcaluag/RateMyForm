Description: 213-csuf-logo-horiz-4c

**CPSC-490: Undergraduate Seminar in Computer Science**
**Project Proposal**

**PLEASE UPLOAD YOUR REPORT TO CANVAS**
**NO PAPER REPORTS**
**1 PER TEAM**
**EVERYONE NEEDS TO UPLOAD**

**Team Member Names: Tracy Tran, Tristan Caluag, Eric Tran, Kyle Ly, Billy Leong**







**Grading Criteria:**

| **Section**            | **Earned Points** | **Possible Points** |
| ---------------------- | ----------------- | ------------------- |
| **Title:**             | <br>              | 5                   |
| **Abstract:**          | <br>              | 10                  |
| **Introduction:**      | <br>              | 15                  |
| **Proposed Solution:** | <br>              | 30                  |
| **Parts List:**        | <br>              | 20                  |
| **RACI Chart:**        | <br>              | 5                   |
| **Schedule:**          | <br>              | 10                  |
| **References:**        | <br>              | 5                   |
| **Total:**             | <br>              | 100                 |

# Professor’s Comments:



# Title : RateMyForm



# Abstract/Project Statement (At Least 5 Complete Sentences) (Tracy):

This project proposes an AI-powered workout form analysis application that provides personalized feedback based on an individual’s body proportions and body structure. Users will record themselves performing an exercise using an accessible device such as a smartphone or camera. The application will use pose-estimation technology to identify body landmarks, track joint positions, calculate joint angles, and analyze movement throughout the exercise [1]–[3]. Similar AI-based exercise correction systems have been developed [3]–[6]. However, this project aims to improve personalization by considering the user’s individual body proportions when evaluating form. Users will enter measurements such as torso length, femur length, lower-leg length, and arm length so the system can use these measurements along with the detected movement. The application will compare the user’s joint angles and movement with research-based exercise guidelines stored in a database while taking the entered body measurements into consideration. It will then identify possible form issues, explain which joint angles or movements may need adjustment, and provide an overall form score. The goal is to make personalized workout guidance more accessible to people who may not have access to or cannot afford a personal trainer. 

# Introduction & Statement of Need (1 to 2 Complete Paragraphs) (Kyle, Tracy):

The rise of social media has made fitness content more accessible than ever, inspiring millions of people to emulate their favorite fitness influencers. However, exercise demonstrations on social media reflect the influencer’s unique body structure and may not suit individuals with different proportions, mobility levels, or biomechanics. Research shows that improper exercise technique can increase the risk of resistance-training injuries and reduce the effectiveness of an exercise [3], [7], [8]. Current fitness resources often promote a single version of “correct” form without accounting for differences in torso length, femur length, limb proportions, or individual movement patterns. As a result, many gym-goers may unknowingly perform exercises in ways that increase injury risk or limit training effectiveness [3], [7], [8]. Therefore, a personalized form-analysis tool is needed to evaluate movement based on an individual's anatomy and body proportions, helping users train more safely, improve exercise efficiency, track progress, and achieve their fitness goals more effectively. 
This problem can be especially difficult for beginners who may not know whether the form they are copying is appropriate for their own body. Instead of expecting every user to match the exact movement of a trainer or fitness influencer, this project will investigate whether individual characteristics such as torso length, femur length, lower-leg length, and arm length can provide additional context when evaluating movement. AI-based exercise systems have already demonstrated that pose estimation can be used to track body movement and provide corrective exercise feedback [1]–[6]. However, this project focuses on combining that type of movement analysis with user-provided body measurements to create more individualized feedback. This could make personalized workout guidance more accessible to people who are learning proper exercise technique or who may not have regular access to a personal trainer [4], [5]. 

# Proposed Solution (2 to 3 Complete Paragraphs, Include Testing Procedure(s)) (Tracy, Kyle):

The application will be designed around equipment that most users already have access to. Users will record themselves performing an exercise with a smartphone or camera and upload the video to the application. A short calibration process will help the user position the camera at an appropriate distance and angle before recording. Users will also enter measurements such as torso length, femur length, lower-leg length, and arm length. User-entered measurements will reduce the system’s reliance on estimating body proportions from appearance alone. Loose or oversized clothing may still obscure body landmarks and affect pose detection, so clothing will be included as a variable during testing. The manually entered measurements will provide the system with additional information about the user’s body proportions and will be used with the movement data collected from the video. This allows the system to evaluate the user’s movement with more information about their individual body structure rather than relying only on the video. 
The proposed solution is a machine-learning and computer-vision application that analyzes exercise form by detecting body landmarks and tracking joint positions throughout an exercise [1]–[3], [9]. Using pose-estimation technology, the system will place virtual nodes on key joints, calculate joint angles [1], [2], [9], and use the body measurements entered by the user to provide personalized feedback tailored to the user’s anatomy rather than comparing everyone to a single standard. To improve accessibility and real-world usability, the system will be tested using exercise footage captured from multiple camera angles, distances, and lighting conditions to determine the range of camera placements under which movement can be reliably analyzed. Testing will be conducted using participants with varying body types, heights, and lifting experience levels while performing supported exercises under normal training conditions. The system’s collected data, including joint positions, movement paths, angle calculations, and feedback accuracy, will be documented and analyzed to refine the application. Additional testing will compare performance across different camera placements and exercise environments to evaluate consistency, reliability, and the effectiveness of the personalized feedback system before final evaluation. 

After the video is analyzed, the application will compare the user’s joint angles and movement with research-based exercise-form guidelines stored in a database. The system will also use the body measurements entered by the user when evaluating the movement. Instead of only labeling the exercise as correct or incorrect, the application will identify specific areas that may need improvement. For example, it may report that the user’s knee, hip, or elbow exceeded an expected angle range during a certain part of the movement. Similar AI-based exercise systems have shown that joint-angle calculations can be used to identify posture differences and provide corrective feedback [1]–[3]. The application will also provide an overall form score, such as 8 out of 10, based on deviations from the exercise-specific joint-angle and movement guidelines. It will explain which movements affected the score so the user understands how the rating was determined. Testing will also include fitted and oversized workout clothing to evaluate whether clothing affects landmark detection, joint-angle calculations, or the consistency of the system’s feedback. This will give users both a simple overall evaluation and specific feedback they can use to improve their form. 

# Detailed Parts List (Insert Spreadsheet or Screenshot) (Billy, Eric):

image

# RACI Chart (Insert Spreadsheet or Screenshot) (Tristan, Eric):

image

# Schedule (Insert Spreadsheet or Screenshot) (Billy, Tristan):

image

# References ([IEEE Format](https://www.myperfectwords.com/blog/citation-styles/ieee-citation-and-format), At Least 5 Sources) (Tracy, Kyle, Eric, Tristan, Billy):

[1] A. Patil, D. Rao, K. Utturwar, T. Shelke, and E. Sarda, “Body posture detection and motion tracking using AI for medical exercises and recommendation system,” *ITM Web of Conferences*, vol. 44, Art. no. 03043, 2022, doi: 10.1051/itmconf/20224403043.
[2] K. M., M. Kandala, V. G. S., V. N., and S. Palaniswamy, “AI-based posture correction, real-time exercise tracking and feedback using pose estimation technique,” in *Proc. 2024 International Conference on Communication, Control, and Intelligent Systems (CCIS)*, Mathura, India, 2024, doi: 10.1109/CCIS63231.2024.10932054.
[3] Prateek, A. Tanvir, and R. Brindha, “AI-powered workout analysis application for posture feedback and repitition grading,” in *Proc. 2024 Second International Conference on Inventive Computing and Informatics (ICICI)*, Bangalore, India, 2024, doi: 10.1109/ICICI62254.2024.00027.
[4] H. Jubair, R. T. Chowdhury, F. Akter, I. S. Sami, N. Nawal, S. T. Arshi, and P. Shil, “Machine learning for real-time exercise correction and injury prevention: A systematic review,” *Authorea Preprints*, 2025, doi: 10.22541/au.174466167.73334339/v1.
[5] J. Medhat and Z. Taha, “AI virtual gym trainer: Using deep learning for real-time exercise guidance and form correction,” in *Proc. 2025 Intelligent Methods, Systems, and Applications (IMSA)*, Giza, Egypt, 2025, doi: 10.1109/IMSA65733.2025.11167265.
[6] J. Singh, A. Parnami, N. Dagar, and M. S. Adhikari, “AI-powered real-time workout pose analysis and correction system,” in *Proc. 2026 5th International Conference on Innovative Practices in Technology and Management (ICIPTM)*, Noida, India, 2026, doi: 10.1109/ICIPTM69057.2026.11465414.
[7] M. J.-Y. Tung, G. A. Lantz, A. D. Lopes, and L. Berglund, “Injuries in weightlifting and powerlifting: An updated systematic review,” *BMJ Open Sport & Exercise Medicine*, vol. 10, no. 4, Art. no. e001884, 2024, doi: 10.1136/bmjsem-2023-001884.
[8] K. Golshani, M. E. Cinque, P. O’Halloran, K. Softness, L. Keeling, and J. R. Macdonell, “Upper extremity weightlifting injuries: Diagnosis and management,” *Journal of Orthopaedics*, vol. 15, no. 1, pp. 24–27, 2018, doi: 10.1016/j.jor.2017.11.005.
[9] M. Ramzan, “Full body detection with OpenCV and MediaPipe,” *Medium*. [Online]. Available:[ https://medium.com/@iamramzan/full-body-detection-with-opencv-and-mediapipe-3b80cfc131ec](https://medium.com/@iamramzan/full-body-detection-with-opencv-and-mediapipe-3b80cfc131ec). [Accessed: Sep. 14, 2026]. 