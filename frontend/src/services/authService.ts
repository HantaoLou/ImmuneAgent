import api from './api';

export interface LoginResponse {
  access_token: string;
  token_type: string;
  username: string;
  role: string;
}

export const login = async (username: string, password: string): Promise<LoginResponse> => {
  try {
    const response = await api.post<LoginResponse>('/api/login', {
      username,
      password,
    });
    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.detail || 'Login failed');
  }
};

export const verifyToken = async (): Promise<boolean> => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      return false;
    }
    await api.get('/api/verify-token');
    return true;
  } catch (error) {
    return false;
  }
};

export const logout = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('username');
};
