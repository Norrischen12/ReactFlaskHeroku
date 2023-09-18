import React, { useState } from 'react';
import './TableDisplay.css';
import IconContainer from './IconContainer';
import DownloadButton from './DownloadButton';
import { useLocation } from 'react-router-dom';
import htmlTableToCsv from 'html-table-to-csv';



/*const data = [
  { product: '0', hf_pk: 523.0, allocation: 1754.607926 },
  { product: '0', hf_pk: 525.0, allocation: 1571.720123 },
  { product: '0', hf_pk: 30388.0, allocation: 5515.796390 },
];*/

function TableDisplay() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const htmlData = queryParams.get('htmlData');

  /*
  const [sortedData, setSortedData] = useState(data);
  const [sortBy, setSortBy] = useState(null);
  const [sortOrder, setSortOrder] = useState('asc');
  const [filterBy, setFilterBy] = useState('');

  function sortData(column) {
    const sortedArray = data.slice().sort((a, b) => {
      if (sortBy === column) {
        return sortOrder === 'asc' ? (a[column] > b[column] ? 1 : -1) : (a[column] > b[column] ? -1 : 1);
      } else {
        return sortOrder === 'asc' ? (a[column] > b[column] ? 1 : -1) : (a[column] > b[column] ? -1 : 1);
      }
    });

  setSortedData(sortedArray);
  setSortBy(column);
  setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
}

  function filterData(value) {
    if (value === 'all') {
      setSortedData(data); // Show all data if 'all' is selected
    } else {
      const filteredData = data.filter((item) => item.product === value);
      setSortedData(filteredData);
    }
    setFilterBy(value);
  }

  return (
    <div className="App">
      <IconContainer />
      <h1>Result Table</h1>
      <div className="controls">
        <label>Filter By Product:</label>
        <select value={filterBy} onChange={(e) => filterData(e.target.value)} className="select-dropdown">
          <option value="all">All</option>
          <option value="0">0</option>
          <option value="1">1</option>
          <option value="2">2</option>
        </select>
        <DownloadButton data={data} /> 
      </div>



      <table id="result-table" style={{ borderRadius: '8px' }} className="intel-font">
        <thead>
          <tr>
            <th onClick={() => sortData('product')} title="Click to sort by product">
              Product
              {sortBy === 'product' && (
                <img src="/sort.svg" alt="Sort Icon" className={`icon ${sortOrder}`} />
              )}
            </th>
            <th onClick={() => sortData('hf_pk')} title="Click to sort by hf_pk">
              hf_pk
              {sortBy === 'hf_pk' && (
                <img src="/sort.svg" alt="Sort Icon" className={`icon ${sortOrder}`} />
              )}
            </th>
            <th onClick={() => sortData('allocation')} title="Click to sort by allocation">
              Allocation
              {sortBy === 'allocation' && (
                <img src="/sort.svg" alt="Sort Icon" className={`icon ${sortOrder}`} />
              )}
            </th>
          </tr>
        </thead>
        
        <tbody>
          {sortedData.map((item, index) => (
            <tr key={index}>
              <td>{item.product}</td>
              <td>{item.hf_pk}</td>
              <td>{item.allocation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
*/
return (
  <div>
    {/* Render the HTML data received from the URL parameter */}
    <div dangerouslySetInnerHTML={{ __html: decodeURIComponent(htmlData) }} />

    {/* Add a button to trigger the download */}
  </div>
);
}




export default TableDisplay;