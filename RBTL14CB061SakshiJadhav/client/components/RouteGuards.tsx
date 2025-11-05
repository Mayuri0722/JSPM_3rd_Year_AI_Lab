import React from 'react';
import { Navigate } from 'react-router-dom';

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'user' | 'admin';
}

export const RequireAuth: React.FC<{
  currentUser: User | null | undefined;
  redirectTo?: string;
  children: React.ReactElement;
}> = ({ currentUser, redirectTo = '/login', children }) => {
  if (!currentUser) return <Navigate to={redirectTo} replace />;
  return children;
};

export const RequireRole: React.FC<{
  currentUser: User | null | undefined;
  roles: Array<'user' | 'admin'>;
  redirectToIfUnauthed?: string;
  redirectToIfForbidden?: string;
  children: React.ReactElement;
}> = ({ currentUser, roles, redirectToIfUnauthed = '/login', redirectToIfForbidden = '/', children }) => {
  if (!currentUser) return <Navigate to={redirectToIfUnauthed} replace />;
  if (!roles.includes(currentUser.role)) return <Navigate to={redirectToIfForbidden} replace />;
  return children;
};
