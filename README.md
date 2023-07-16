# AGH-BirdChirpRecognition
A project for Amrita Green Hackathon - ChirpSearch  
  
This project is a solution to a problem statement that was provided in the hackathon where we had to find out a way in recognizing the birds that are found in Amrita Vishwa Vidyapeetham Coimbatore Campus based on their chirps, cries, and sounds they make.  

This project uses the help of a CNN model to predict the bird with the respective bird cry that is recorded using the mobile application app.  
This model is trained by taking the MFCC ( Mel-Frequency Cepstral Coefficients ) along with doing Fourier Transform(Time domain to Frequency Domain) for each of the bird sounds that is being trained so that the corresponding bird cry can be classified and labeled based on the specific features collected.  
Once the recording is sent to the model, the model recognizes the bird's chirp and displays the bird's name in the mobile application. 
