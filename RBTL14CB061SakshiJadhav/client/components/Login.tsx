import React, { useState } from 'react';
import { login } from '../services/authService';
import { Link } from 'react-router-dom';

interface Props {
  onSuccess: (payload: { token: string; user: { id: string; name: string; email: string; role: 'user' | 'admin' } }) => void;
}

const Login: React.FC<Props> = ({ onSuccess }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await login(email, password);
      onSuccess(data);
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto bg-gray-800/50 p-6 rounded-lg border border-gray-700/50">
      <h2 className="text-xl font-semibold mb-4">Login</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="email"
          placeholder="Email"
          className="w-full p-2 bg-gray-800 border border-gray-600 rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full p-2 bg-gray-800 border border-gray-600 rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error && <div className="text-red-400 text-sm">{error}</div>}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2 bg-blue-600 rounded hover:bg-blue-700 disabled:bg-gray-600"
        >
          {loading ? 'Logging in...' : 'Login'}
        </button>
      </form>
      <div className="mt-3 text-sm text-gray-400">
        Don't have an account?{' '}
        <Link className="text-blue-400 hover:text-blue-300" to="/signup">Sign up</Link>
      </div>
    </div>
  );
};

export default Login;
