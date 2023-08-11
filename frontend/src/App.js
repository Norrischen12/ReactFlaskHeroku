import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import App from './AppPage';
import TableDisplay from './TableDisplay';

function RootApp() {
  return (
    <Router>
      <Routes>
        <Route exact path="/" element={<App/>} />

      </Routes>
    </Router>
  );
}

export default RootApp;