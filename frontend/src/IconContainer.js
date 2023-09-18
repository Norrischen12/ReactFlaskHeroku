import React from 'react';
import './IconContainer.css'; // Create a separate CSS file for styling the icon container

const IconContainer = () => {
  return (
    <div className="icon-container">
      {/* Your icons and other elements */}
      <img src="/logo_1.png" alt="Icon 1" className="logo_icon" />
      <img src="/logo_2.png" alt="Icon 2" className="logo_con" />
    </div>
  );
};

export default IconContainer;
