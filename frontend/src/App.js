import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import AppPage from './AppPage';
import LoadingCircle from './LoadingCircle';
import TableDisplay from './TableDisplay';

function RootApp() {
  return (
    <Router>
      <Routes>
        <Route exact path="/" element={<AppPage/>} />
        <Route exact path="/table" element={<TableDisplay/>} />
      </Routes>
    </Router>
  );
}

export default RootApp;