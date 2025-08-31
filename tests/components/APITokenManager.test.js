import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import APITokenManager from '../../components/settings/APITokenManager';

// Mock fetch globally
global.fetch = vi.fn();

describe('APITokenManager Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock successful API responses
    fetch.mockImplementation((url) => {
      if (url.includes('/api/settings/tokens')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            tokens: [
              {
                _id: 'token1',
                name: 'Test Token',
                maskedToken: 'test-tok...ken1',
                permissions: [
                  { model: 'User', actions: ['read', 'create'] }
                ],
                rateLimit: { requests: 1000, window: 3600 },
                usage: { totalRequests: 50, lastUsed: new Date() },
                isActive: true,
                createdAt: new Date(),
                expiresAt: new Date(Date.now() + 86400000) // 1 day from now
              }
            ]
          })
        });
      }
      
      if (url.includes('/api/models')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            models: [
              { name: 'User' },
              { name: 'Media' },
              { name: 'Settings' }
            ]
          })
        });
      }
      
      return Promise.reject(new Error('Unknown URL'));
    });
  });

  it('should render the API token manager interface', async () => {
    render(<APITokenManager />);
    
    // Wait for the component to load
    await waitFor(() => {
      expect(screen.getByText('API Token Management')).toBeInTheDocument();
    });
    
    expect(screen.getByText('Create and manage API tokens for programmatic access')).toBeInTheDocument();
    expect(screen.getByText('Create Token')).toBeInTheDocument();
  });

  it('should display tokens when loaded', async () => {
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Token')).toBeInTheDocument();
    });
    
    expect(screen.getByText('test-tok...ken1')).toBeInTheDocument();
    expect(screen.getByText('active')).toBeInTheDocument();
    expect(screen.getByText('Usage: 50 requests')).toBeInTheDocument();
  });

  it('should show create token modal when create button is clicked', async () => {
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Create Token')).toBeInTheDocument();
    });
    
    const createButton = screen.getByText('Create Token');
    fireEvent.click(createButton);
    
    await waitFor(() => {
      expect(screen.getByText('Create API Token')).toBeInTheDocument();
    });
    
    expect(screen.getByLabelText('Token Name *')).toBeInTheDocument();
    expect(screen.getByLabelText('Description')).toBeInTheDocument();
    expect(screen.getByText('Add Permission')).toBeInTheDocument();
  });

  it('should handle loading state', () => {
    // Mock fetch to never resolve to simulate loading
    fetch.mockImplementation(() => new Promise(() => {}));
    
    render(<APITokenManager />);
    
    expect(screen.getByText('Loading tokens...')).toBeInTheDocument();
  });

  it('should handle empty state', async () => {
    // Mock empty response
    fetch.mockImplementation((url) => {
      if (url.includes('/api/settings/tokens')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ tokens: [] })
        });
      }
      
      if (url.includes('/api/models')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ models: [] })
        });
      }
      
      return Promise.reject(new Error('Unknown URL'));
    });
    
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('No API Tokens')).toBeInTheDocument();
    });
    
    expect(screen.getByText('Create your first API token to get started with programmatic access.')).toBeInTheDocument();
  });

  it('should handle API errors', async () => {
    // Mock error response
    fetch.mockImplementation(() => 
      Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ error: 'Failed to fetch tokens' })
      })
    );
    
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch tokens')).toBeInTheDocument();
    });
  });

  it('should toggle inactive tokens visibility', async () => {
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Show inactive tokens')).toBeInTheDocument();
    });
    
    const toggleCheckbox = screen.getByLabelText('Show inactive tokens');
    fireEvent.click(toggleCheckbox);
    
    // Should trigger a new fetch with includeInactive=true
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/settings/tokens?includeInactive=true');
    });
  });

  it('should show token permissions correctly', async () => {
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Permissions (1)')).toBeInTheDocument();
    });
    
    expect(screen.getByText('User: read, create')).toBeInTheDocument();
  });

  it('should display rate limit information', async () => {
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Rate Limit')).toBeInTheDocument();
    });
    
    expect(screen.getByText('1000/3600s')).toBeInTheDocument();
  });

  it('should handle token creation form validation', async () => {
    render(<APITokenManager />);
    
    await waitFor(() => {
      expect(screen.getByText('Create Token')).toBeInTheDocument();
    });
    
    const createButton = screen.getByText('Create Token');
    fireEvent.click(createButton);
    
    await waitFor(() => {
      expect(screen.getByText('Create API Token')).toBeInTheDocument();
    });
    
    // Try to submit without required fields
    const submitButton = screen.getByText('Create Token');
    expect(submitButton).toBeDisabled();
  });
});