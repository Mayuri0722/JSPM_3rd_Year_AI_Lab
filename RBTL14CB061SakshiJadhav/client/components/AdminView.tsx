import React, { useEffect, useState } from 'react';

interface Props {
  token: string;
}

const AdminView: React.FC<Props> = ({ token }) => {
  const [status, setStatus] = useState<'idle' | 'ok' | 'error'>('idle');
  const [message, setMessage] = useState<string>('');

  useEffect(() => {
    const checkAdmin = async () => {
      try {
        const res = await fetch((import.meta.env.VITE_API_BASE || 'http://localhost:4000') + '/api/auth/admin-only', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          setStatus('error');
          setMessage(data.message || 'Forbidden');
        } else {
          setStatus('ok');
          setMessage('Welcome, Admin');
        }
      } catch (e: any) {
        setStatus('error');
        setMessage(e?.message || 'Network error');
      }
    };
    if (token) checkAdmin();
  }, [token]);

  return (
    <div className="max-w-2xl mx-auto bg-gray-800/50 p-6 rounded-lg border border-gray-700/50">
      <h2 className="text-xl font-semibold mb-4">Admin Dashboard</h2>
      {status === 'idle' && <div className="text-gray-400">Checking access…</div>}
      {status === 'ok' && <div className="text-green-400">{message}</div>}
      {status === 'error' && <div className="text-red-400">{message}</div>}
      <div className="mt-4 text-sm text-gray-300">Only users with role <code>admin</code> can access this page.</div>
    </div>
  );
};

export default AdminView;
