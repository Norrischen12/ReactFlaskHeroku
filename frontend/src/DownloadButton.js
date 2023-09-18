import React from 'react';
import * as XLSX from 'xlsx';
import './DownloadButton.css';
import Download from './Download_icon.svg';

const DownloadButton = ({ data }) => {
  const handleDownload = () => {
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
    XLSX.writeFile(wb, 'table_data.xlsx');
  };

  return (
    <button className="download-button" onClick={handleDownload}>
      <img src={Download} alt="Download Icon" className="download-icon" />
      Download
    </button>
  );
};

export default DownloadButton;

