import React, { useState, useEffect } from 'react';
import axios from 'axios';

function TableDisplay() {
  const [htmlTable, setHtmlTable] = useState('');

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await axios.post('http://localhost:5000/members', {
          // Provide the necessary data here that your backend expects
          // Provide the selected file object
        });

        // Extract the HTML table markup from the response
        const { html_table } = response.data;

        setHtmlTable(html_table);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    }

    fetchData();
  }, []);

  return (
    <div>
      {/* Render the received HTML table markup */}
      <div dangerouslySetInnerHTML={{ __html: htmlTable }} />
    </div>
  );
}

export default TableDisplay;
