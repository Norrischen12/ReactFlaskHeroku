import React, { useState } from 'react';
import logo from "./logo.svg";
import document_upload from "./document_upload.svg";
import doc from "./doc.svg";
import delete_icon from "./delete.svg";
import { Link, Navigate } from 'react-router-dom';
import * as S from "./App.styles";
import LoadingCircle from './LoadingCircle';
import { useNavigate } from 'react-router-dom';


function FileVerification(fileName) {
  const fileExtension = fileName.split('.').pop().toLowerCase();
  return fileExtension === 'csv' || fileExtension === 'xlsx';
}

function App() {
  const navigate = useNavigate();
  const [selectedDate, setSelectedDate] = useState(null); // Add state for selectedDate
  const [selectedFile, setSelectedFile] = useState(null); // Add state for selectedFile
  const [htmlTable, setHtmlTable] = useState("");
  const [isLoading, setIsLoading] = useState(false);



  const handleDateChange = (event) => {
    setSelectedDate(event.target.value); // Update selectedDate state on date change
    console.log("Date Change to " + typeof selectedDate)
  };
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]); // Update selectedFile state when file input changes
    console.log("File changed to :" + typeof selectedFile)
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    // Create a data object to send the data
    const formData = new FormData();

    formData.append("arg1", selectedDate);
    formData.append("arg2", selectedFile);

    try {
      // Make the API request to your Flask server
      const response = await fetch("http://localhost:5000/members", {
        method: "POST",
        mode: "cors",
        body: formData,
      });

      if (response.ok) {
        // Process the successful response here
        const resultData = await response.json();
        console.log("Result Data:", resultData);
        // Update your React state or UI with the resultData
        setHtmlTable(resultData.html_table);

        const htmlData = encodeURIComponent(resultData.html_table)
        console.log('Navigating to /table')
        navigate(`/table?htmlData=${htmlData}`)
      } else {
        // Handle error response
        const errorData = await response.json();
        console.error("Error:", errorData.error);
        // Display an error message to the user
        
      }
    } catch (error) {
      // Handle any network or other errors
      console.error("Error:", error);
      // Display an error message to the user
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <S.PageContanier>
      <S.LogoContanier>
        <img src={logo} className="App-logo" alt="logo" />
      </S.LogoContanier>
      <S.FormContanier>
        <S.FormTitle>Select Date</S.FormTitle>
        <S.FormInput
          type="date"
          value={selectedDate}
          onChange={handleDateChange}
          placeholder="Choose a date..."
          // onFocus={(e) => (e.target.type = "date")}
          // onBlur={(e) => (e.target.type = "text")}
        />
      </S.FormContanier>
      
      <S.FormContanier>
        <S.FormTitleWrapper>
          <S.FormTitle>Upload File</S.FormTitle>
          <S.FormSubtitle>
            {" "}
            (Please note that it should be .xlsx file format)
          </S.FormSubtitle>
        </S.FormTitleWrapper>
        <S.CustomFormFileInput>
          <S.FormFileInput
            id="upload_file"
            type="file"
            accept=".xlsx"
            onChange={handleFileChange}
          />
          <S.IconWrapper width="120px" height="120px">
            <img src={document_upload} alt="logo" />
          </S.IconWrapper>
          <S.FileInputTitle>.XLSX</S.FileInputTitle>
          <S.FileInputSubtitle>
            Click to browse or drag and drop your files here
          </S.FileInputSubtitle>
        </S.CustomFormFileInput>
      </S.FormContanier>

      <S.InfoContainer>
        <S.IconWrapper width="65px" height="65px">
          <img src={doc} alt="logo" />
        </S.IconWrapper>
        <S.InfoWrapper>
          <S.FormTitle>NacNac_Data.xlsx</S.FormTitle>
          <S.StorageInfo>89/124 KB</S.StorageInfo>
          <S.ProgressBar />
          <S.ProgressBarDynamic />
        </S.InfoWrapper>
        <S.IconButton width="42px" height="42px">
          <img src={delete_icon} alt="logo" />
        </S.IconButton>
      </S.InfoContainer>
      <S.ButtonWrapper>
        <S.SubmitButton onClick={handleSubmit}>
          {isLoading ? <LoadingCircle /> : "Run Script"}
        </S.SubmitButton>
      </S.ButtonWrapper>
      <div>
        {/* Render the HTML table */}
        <div dangerouslySetInnerHTML={{ __html: htmlTable }} />
      </div>
    </S.PageContanier>
  );
}

export default App;
