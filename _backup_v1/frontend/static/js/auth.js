const Auth = {
  TOKEN_KEY: 'rag_medical_token',
  USER_KEY: 'rag_medical_user',

  getToken() {
    return localStorage.getItem(this.TOKEN_KEY);
  },

  getUser() {
    try {
      return JSON.parse(localStorage.getItem(this.USER_KEY));
    } catch {
      return null;
    }
  },

  isLoggedIn() {
    return !!this.getToken();
  },

  saveLogin(token, user) {
    localStorage.setItem(this.TOKEN_KEY, token);
    localStorage.setItem(this.USER_KEY, JSON.stringify(user));
  },

  logout() {
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.USER_KEY);
    window.location.href = '/login';
  },

  requireAuth() {
    if (!this.isLoggedIn()) {
      window.location.href = '/login';
      return false;
    }
    return true;
  },

  authHeaders() {
    const token = this.getToken();
    if (!token) return {};
    return { 'Authorization': `Bearer ${token}` };
  },

  async apiFetch(url, options = {}) {
    const headers = { ...options.headers, ...this.authHeaders() };
    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
      this.logout();
      throw new Error('Sesión expirada. Inicia sesión nuevamente.');
    }

    return response;
  },

  getInitials(name) {
    if (!name) return '?';
    return name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
  }
};
