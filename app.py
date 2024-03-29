import streamlit as st
from tensorflow.keras.models import load_model
from files_upload import FilesUpload

def main():

    st.title("IQGateway (Malaria Detector Assignment) ~ Shubham Raj")
    acitvity = ['About Data', 'Prediction', 'About me']
    choice = st.sidebar.selectbox('Chose An Activity', acitvity)

    if choice == 'About Data':
        st.subheader("Kaggle DATA SOURCE")
        st.text("The dataset contains 2 folders - Infected - Uninfected")
        st.text("Acknowledgements This Dataset is taken from the official NIH Website: ")
        st.markdown("https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets")
        #print(st.__version__)


    if choice == 'Prediction':

        #image_shape = (130, 130, 3)
        #model = load_model('./models/malaria_detector.h5')
        # Taking Instance of Class 'FilesUpload' by calling "from files_upload import FilesUpload"
        files_upload = FilesUpload()
        img = files_upload.run()
        if st.button("Predict"):
            st.text('Wait...Model is being loaded!')
            model = load_model(r'C:\Users\SHUBHAM RAJ\Downloads\Malaria_classification\malaria_detector.h5')
            st.success("Model Loaded")
            st.text('Wait...')
            if model.predict(img)[0][0] > 0.5:
                st.text("Uninfected")
                st.text("Probability: {}".format(model.predict(img)[0][0]))
            else:
                st.text("If irrelevent image is uploaded then model will assume it is infected")
                st.text("Infected/Parasitized")
                st.text("Probability: {}".format(model.predict(img)[0][0]))



    if choice == 'About me':
        st.subheader("Malaria Dectection Shubham Raj ~ Assignment iqGateway")
        st.info("sr6760.sr@gmail.com")


if __name__ == '__main__':
    main()

