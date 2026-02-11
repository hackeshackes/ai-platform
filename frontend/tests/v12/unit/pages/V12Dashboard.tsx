import React, { useEffect, useState } from 'react';
import { getProjects, getModels } from '../api/v12API';

interface Project {
  id: string;
  name: string;
  status: string;
}

interface Model {
  id: string;
  name: string;
  provider: string;
}

export const V12Dashboard: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [projectsData, modelsData] = await Promise.all([
          getProjects(),
          getModels(),
        ]);
        setProjects(projectsData);
        setModels(modelsData);
      } catch (error) {
        console.error('Failed to load data:', error);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  return (
    <div data-testid="v12-dashboard" className="v12-dashboard">
      <h1 data-testid="dashboard-title">AI Platform Dashboard</h1>
      <div className="v12-dashboard-content">
        <section className="projects-section">
          <h2>Projects</h2>
          {loading ? (
            <p>Loading projects...</p>
          ) : (
            <div className="project-list">
              {projects.map((project) => (
                <div key={project.id} className="project-card">
                  <h3>{project.name}</h3>
                  <span className={`status status-${project.status}`}>
                    {project.status}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
        <section className="models-section">
          <h2>Available Models</h2>
          {loading ? (
            <p>Loading models...</p>
          ) : (
            <div className="model-list">
              {models.map((model) => (
                <div key={model.id} className="model-card">
                  <h3>{model.name}</h3>
                  <span className="provider">{model.provider}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
