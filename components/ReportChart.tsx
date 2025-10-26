
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ChartDataPoint } from '../types';

interface ReportChartProps {
  data: ChartDataPoint[];
}

const ReportChart: React.FC<ReportChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="time" stroke="#94a3b8" />
        <YAxis stroke="#94a3b8" />
        <Tooltip 
            contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #334155',
                color: '#f1f5f9'
            }} 
            labelStyle={{ color: '#cbd5e1' }}
        />
        <Legend wrapperStyle={{ color: '#f1f5f9' }} />
        <Line 
            type="monotone" 
            dataKey="wpm" 
            name="Words Per Minute" 
            stroke="#2dd4bf" 
            strokeWidth={2}
            activeDot={{ r: 8, fill: '#2dd4bf', stroke: '#1e293b', strokeWidth: 2 }} 
            dot={{ r: 4, fill: '#2dd4bf' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ReportChart;
