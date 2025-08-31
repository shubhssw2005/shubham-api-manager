import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import IncidentManager from '../../lib/incident-response/IncidentManager.js';

describe('IncidentManager', () => {
  let incidentManager;
  let mockConfig;

  beforeEach(() => {
    mockConfig = {
      pagerDutyApiKey: 'test-api-key',
      serviceKey: 'test-service-key',
      slackWebhookUrl: 'https://hooks.slack.com/test'
    };
    
    incidentManager = new IncidentManager(mockConfig);
    
    // Mock external API calls
    vi.mock('axios');
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('createIncident', () => {
    it('should create a new incident with required fields', async () => {
      const incidentData = {
        title: 'Test Incident',
        description: 'Test incident description',
        severity: 'critical',
        service: 'api'
      };

      const incident = await incidentManager.createIncident(incidentData);

      expect(incident).toMatchObject({
        title: 'Test Incident',
        description: 'Test incident description',
        severity: 'critical',
        service: 'api',
        status: 'triggered'
      });
      expect(incident.id).toBeDefined();
      expect(incident.createdAt).toBeInstanceOf(Date);
      expect(incident.timeline).toHaveLength(0);
    });

    it('should generate unique incident IDs', async () => {
      const incidentData = {
        title: 'Test Incident',
        description: 'Test description',
        service: 'api'
      };

      const incident1 = await incidentManager.createIncident(incidentData);
      const incident2 = await incidentManager.createIncident(incidentData);

      expect(incident1.id).not.toBe(incident2.id);
    });

    it('should store incident in active incidents map', async () => {
      const incidentData = {
        title: 'Test Incident',
        service: 'api'
      };

      const incident = await incidentManager.createIncident(incidentData);

      expect(incidentManager.activeIncidents.has(incident.id)).toBe(true);
      expect(incidentManager.activeIncidents.get(incident.id)).toBe(incident);
    });
  });

  describe('updateIncident', () => {
    let incident;

    beforeEach(async () => {
      incident = await incidentManager.createIncident({
        title: 'Test Incident',
        service: 'api'
      });
    });

    it('should update incident properties', async () => {
      const updates = {
        status: 'investigating',
        description: 'Updated description'
      };

      const updatedIncident = await incidentManager.updateIncident(incident.id, updates);

      expect(updatedIncident.status).toBe('investigating');
      expect(updatedIncident.description).toBe('Updated description');
      expect(updatedIncident.updatedAt).toBeInstanceOf(Date);
    });

    it('should add timeline entry on update', async () => {
      const updates = {
        status: 'investigating',
        user: 'test-user'
      };

      const updatedIncident = await incidentManager.updateIncident(incident.id, updates);

      expect(updatedIncident.timeline).toHaveLength(1);
      expect(updatedIncident.timeline[0]).toMatchObject({
        action: 'investigating',
        user: 'test-user'
      });
    });

    it('should throw error for non-existent incident', async () => {
      await expect(
        incidentManager.updateIncident('non-existent-id', { status: 'resolved' })
      ).rejects.toThrow('Incident non-existent-id not found');
    });
  });

  describe('resolveIncident', () => {
    let incident;

    beforeEach(async () => {
      incident = await incidentManager.createIncident({
        title: 'Test Incident',
        service: 'api'
      });
    });

    it('should resolve incident with resolution details', async () => {
      const resolution = {
        description: 'Fixed by restarting service',
        user: 'engineer'
      };

      const resolvedIncident = await incidentManager.resolveIncident(incident.id, resolution);

      expect(resolvedIncident.status).toBe('resolved');
      expect(resolvedIncident.resolvedAt).toBeInstanceOf(Date);
      expect(resolvedIncident.resolution).toBe(resolution);
    });

    it('should add resolution to timeline', async () => {
      const resolution = {
        description: 'Service restarted',
        user: 'engineer'
      };

      const resolvedIncident = await incidentManager.resolveIncident(incident.id, resolution);

      const resolutionEntry = resolvedIncident.timeline.find(entry => entry.action === 'resolved');
      expect(resolutionEntry).toBeDefined();
      expect(resolutionEntry.description).toBe('Service restarted');
      expect(resolutionEntry.user).toBe('engineer');
    });

    it('should remove incident from active incidents', async () => {
      const resolution = {
        description: 'Fixed',
        user: 'engineer'
      };

      await incidentManager.resolveIncident(incident.id, resolution);

      expect(incidentManager.activeIncidents.has(incident.id)).toBe(false);
    });
  });

  describe('escalation', () => {
    beforeEach(() => {
      // Mock time for consistent testing
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should escalate critical incidents after threshold', async () => {
      const incident = await incidentManager.createIncident({
        title: 'Critical Database Issue',
        severity: 'critical',
        service: 'database'
      });

      // Advance time past escalation threshold (10 minutes)
      vi.advanceTimersByTime(11 * 60 * 1000);

      await incidentManager.checkEscalation(incident);

      expect(incident.escalated).toBe(true);
      expect(incident.escalatedAt).toBeInstanceOf(Date);
    });

    it('should not escalate incidents before threshold', async () => {
      const incident = await incidentManager.createIncident({
        title: 'Warning Issue',
        severity: 'warning',
        service: 'api'
      });

      // Advance time but not past threshold
      vi.advanceTimersByTime(5 * 60 * 1000);

      await incidentManager.checkEscalation(incident);

      expect(incident.escalated).toBeUndefined();
    });
  });

  describe('auto-response', () => {
    it('should execute matching auto-response rules', async () => {
      const mockAction = vi.fn();
      
      // Add a test auto-response rule
      incidentManager.autoResponseRules.set('database', [{
        name: 'test-rule',
        condition: { description: /connection.*pool/i },
        action: mockAction
      }]);

      const incident = await incidentManager.createIncident({
        title: 'Database Issue',
        description: 'Connection pool exhausted',
        service: 'database'
      });

      expect(mockAction).toHaveBeenCalledWith(incident);
    });

    it('should not execute non-matching auto-response rules', async () => {
      const mockAction = vi.fn();
      
      incidentManager.autoResponseRules.set('database', [{
        name: 'test-rule',
        condition: { description: /different.*pattern/i },
        action: mockAction
      }]);

      await incidentManager.createIncident({
        title: 'Database Issue',
        description: 'Connection pool exhausted',
        service: 'database'
      });

      expect(mockAction).not.toHaveBeenCalled();
    });
  });

  describe('severity mapping', () => {
    it('should map severity to PagerDuty correctly', () => {
      expect(incidentManager.mapSeverityToPagerDuty('critical')).toBe('critical');
      expect(incidentManager.mapSeverityToPagerDuty('warning')).toBe('warning');
      expect(incidentManager.mapSeverityToPagerDuty('info')).toBe('info');
      expect(incidentManager.mapSeverityToPagerDuty('unknown')).toBe('warning');
    });
  });

  describe('duration formatting', () => {
    it('should format duration correctly', () => {
      const start = new Date('2023-01-01T10:00:00Z');
      const end = new Date('2023-01-01T11:30:00Z');
      
      const formatted = incidentManager.formatDuration(start, end);
      expect(formatted).toBe('1h 30m');
    });

    it('should handle ongoing incidents', () => {
      const start = new Date();
      const formatted = incidentManager.formatDuration(start, null);
      expect(formatted).toBe('Ongoing');
    });

    it('should format minutes only for short durations', () => {
      const start = new Date('2023-01-01T10:00:00Z');
      const end = new Date('2023-01-01T10:15:00Z');
      
      const formatted = incidentManager.formatDuration(start, end);
      expect(formatted).toBe('15m');
    });
  });
});