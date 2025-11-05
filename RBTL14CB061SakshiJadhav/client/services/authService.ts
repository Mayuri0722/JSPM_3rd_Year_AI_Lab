const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:4000';

export interface AuthResponse {
  token: string;
  user: { id: string; name: string; email: string; role: 'user' | 'admin' };
}

export async function signup(name: string, email: string, password: string, role: 'user' | 'admin' = 'user'): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/api/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password, role })
  });
  if (!res.ok) throw new Error((await res.json()).message || 'Signup failed');
  return res.json();
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  if (!res.ok) throw new Error((await res.json()).message || 'Login failed');
  return res.json();
}

export async function getMe(token: string) {
  const res = await fetch(`${API_BASE}/api/auth/me`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!res.ok) throw new Error((await res.json()).message || 'Fetch user failed');
  return res.json();
}
