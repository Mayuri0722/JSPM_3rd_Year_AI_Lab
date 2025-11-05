
import React from 'react';

const Loader: React.FC = () => (
  <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" role="status">
    <span className="sr-only">Loading...</span>
  </div>
);

export default Loader;
