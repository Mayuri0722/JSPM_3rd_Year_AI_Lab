import React, { useState } from 'react';
import { signup } from '../services/authService';
import { Link } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';

interface Props {
  onSuccess?: (payload: { token: string; user: { id: string; name: string; email: string; role: 'user' | 'admin' } }) => void;
}

const Signup: React.FC<Props> = ({ onSuccess }) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'user' | 'admin'>('user');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signup(name, email, password, role);
      // After successful signup, go to login page
      navigate('/login');
    } catch (err: any) {
      setError(err.message || 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto bg-gray-800/50 p-6 rounded-lg border border-gray-700/50">
      <h2 className="text-xl font-semibold mb-4">Create Account</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="text"
          placeholder="Name"
          className="w-full p-2 bg-gray-800 border border-gray-600 rounded"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
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
        <select
          className="w-full p-2 bg-gray-800 border border-gray-600 rounded"
          value={role}
          onChange={(e) => setRole(e.target.value as 'user' | 'admin')}
        >
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
        {error && <div className="text-red-400 text-sm">{error}</div>}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2 bg-blue-600 rounded hover:bg-blue-700 disabled:bg-gray-600"
        >
          {loading ? 'Creating...' : 'Sign up'}
        </button>
      </form>
      <div className="mt-3 text-sm text-gray-400">
        Already have an account?{' '}
        <Link className="text-blue-400 hover:text-blue-300" to="/login">Login</Link>
      </div>
    </div>
  );
};

export default Signup;
