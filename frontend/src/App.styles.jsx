import styled from "styled-components";
import calendar from "./calendar.svg";

export const PageContanier = styled.div`
  display: flex;
  flex-direction: column;
  padding: 32px 50px;
  gap: 22px;
`;

// Logo

export const LogoContanier = styled.div`
  display: flex;
  width: 100%;
  height: 80px;
  justify-content: center;
`;

// Form

export const FormContanier = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: 8px;
`;

// // Form Title

export const FormTitleWrapper = styled.div`
  display: flex;
`;

export const FormTitle = styled.p`
  margin: 0;
  padding: 0;
  font-family: Inter;
  font-size: 16px;
  font-style: normal;
  font-weight: 700;
`;

export const FormSubtitle = styled.p`
  margin: 0;
  padding: 0;
  color: #7b7b7b;
  font-size: 15px;
  font-weight: 400;
`;

// // Form Input

export const FormInput = styled.input.attrs({ type: "date" })`
  max-width: 360px;
  padding: 8px;
  border-radius: 8px;
  border: 2px solid #2c78ae;
  background: #fff;
  font-family: Inter;
  &::-webkit-calendar-picker-indicator {
    opacity: 1;
    display: block;
    width: 30px;
    height: 30px;
    background: url(${calendar}) no-repeat;
    border-width: thin;
  }
`;

export const FormFileInput = styled.input.attrs({ type: "file" })`
  display: none;
`;

export const CustomFormFileInput = styled.label`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 75px 0px;
  width: 100%;
  border-radius: 8px;
  border: 2px dashed #2c78ae;
  background: rgba(132, 157, 174, 0.09);
  cursor: pointer;
`;

export const FileInputTitle = styled.p`
  display: flex;
  margin: 20px 0px 0px 0px;
  padding: 0;
  color: #737373;
  font-size: 16px;
  font-weight: 700;
  line-height: 150%;
`;

export const FileInputSubtitle = styled.p`
  display: flex;
  margin: 0;
  padding: 0;
  color: #737373;
  font-size: 16px;
  font-weight: 400;
  line-height: 150%;
`;

export const IconWrapper = styled.div`
  width: ${(props) => props.width};
  height: ${(props) => props.height};
`;

export const IconButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 10px;
  width: ${(props) => props.width};
  height: ${(props) => props.height};
  border: none;
  background-color: ${(props) => props.backgroundColor || "transparent"};
  color: ${(props) => props.color || "#000"};
  border-radius: 50%;
  cursor: pointer;
  transition: background-color 0.3s, color 0.3s;

  &:hover {
    background-color: #eee;
    // background-color: ${(props) =>
      props.hoverBackgroundColor || "transparent"};
    // color: ${(props) => props.hoverColor || props.color || "#000"};
  }
`;

// Document Collection

export const InfoContainer = styled.div`
  display: flex;
  align-items: center;
  padding: 30px;
  border-radius: 5px;
  box-shadow: 0px 0px 4px 0px rgba(0, 0, 0, 0.5);
  gap: 16px;
`;

export const InfoWrapper = styled.div`
  display: flex;
  flex-direction: column;
  width: 85%;
  margin-right: 40px;
`;

export const StorageInfo = styled.p`
  margin: 0;
  padding: 10px 0px 4px 0px;
  color: #9f9f9f;
  font-size: 12px;
  font-weight: 400;
  line-height: 150%;
`;

export const ProgressBar = styled.div`
  height: 10px;
  border-radius: 20px;
  background-color: #d4d4d4;
`;

export const ProgressBarDynamic = styled.div`
  height: 10px;
  border-radius: 20px;
  background-color: #3ea6f0;
  margin-top: -10px;
  width: 60%;
`;

// Button

export const ButtonWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const SubmitButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 160px;
  height: 52px;
  background-color: #2c78ae;
  color: #fff;
  font-size: 16px;
  font-weight: 700;
  border: none;
  border-radius: 6px;
`;
