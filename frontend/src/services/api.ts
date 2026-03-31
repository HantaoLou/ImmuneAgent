import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      if (token && config.url !== '/api/login') {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    // When sending FormData, remove Content-Type so the browser/XHR can set the
    // correct multipart/form-data header WITH the boundary string.
    // The instance default "application/json" causes axios to serialize FormData
    // as JSON, resulting in an empty body on the server side.
    // Must use AxiosHeaders.delete() — the JS `delete` operator is a no-op on
    // the AxiosHeaders proxy object.
    if (typeof FormData !== 'undefined' && config.data instanceof FormData) {
      config.headers.delete('Content-Type');
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }
      }
    }
    if (error.response) {
      // Server responded with error status
      console.error(
        `[API] Response error ${error.config?.method?.toUpperCase()} ${error.config?.url}`,
        error.response.status,
        error.response.data,
      );
    } else if (error.request) {
      // Request was sent but no response received (timeout / network error / CORS)
      console.error(
        `[API] No response for ${error.config?.method?.toUpperCase()} ${error.config?.url}`,
        `code=${error.code} message="${error.message}"`,
      );
    } else {
      // Request setup error
      console.error('[API] Request setup error:', error.message);
    }
    return Promise.reject(error);
  }
);

export default api;
