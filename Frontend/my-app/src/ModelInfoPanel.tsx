import { Brain, TreePine, Zap, CheckCircle, XCircle, Database, Leaf } from 'lucide-react'
import type { ModelsResponse } from '../types'

const MODEL_ICONS: Record<string, React.ReactNode> = {
  random_forest: <TreePine size={14} />,
  xgboost: <Zap size={14} />,
  neural_network: <Brain size={14} />,
  lightgbm: <Leaf size={14} />,
}

const MODEL_COLORS: Record<string, string> = {
  random_forest: '#4CAF50',
  xgboost: '#FF9800',
  neural_network: '#4a9eff',
  lightgbm: '#9b59b6',
}

interface ModelInfoPanelProps {
  modelsData: ModelsResponse | null
}

export default function ModelInfoPanel({ modelsData }: ModelInfoPanelProps) {
  if (!modelsData) {
    return (
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="section-title">Model Status</span>
        </div>
        <div className="text-[#5c5868] text-xs animate-pulse">Loading model info...</div>
      </div>
    )
  }

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="section-title">Model Status</span>
        {modelsData.training_samples && (
          <div className="flex items-center gap-1 text-[10px] text-[#5c5868]">
            <Database size={10} />
            {modelsData.training_samples.toLocaleString()} samples
          </div>
        )}
      </div>

      <div className="flex flex-col gap-2">
        {modelsData.models.map((model) => {
          const color = MODEL_COLORS[model.id] ?? '#4a9eff'
          const icon = MODEL_ICONS[model.id]

          return (
            <div key={model.id} className="flex items-center gap-3 py-1.5 border-b border-[#2a2a3a] last:border-0">
              {/* Status indicator */}
              <div style={{ color }}>
                {icon}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-[#e8e6f0]">{model.display_name}</span>
                  {model.loaded
                    ? <CheckCircle size={10} className="text-green-400" />
                    : <XCircle size={10} className="text-red-400" />}
                </div>
                <p className="text-[10px] text-[#5c5868] truncate">{model.description}</p>
              </div>

              {/* Metrics */}
              {(model.accuracy !== null || model.auc !== null) ? (
                <div className="flex flex-col items-end gap-0.5 shrink-0">
                  {model.accuracy !== null && (
                    <span className="text-[10px]" style={{ color }}>
                      {(model.accuracy * 100).toFixed(1)}% acc
                    </span>
                  )}
                  {model.auc !== null && (
                    <span className="text-[10px] text-[#5c5868]">
                      {model.auc.toFixed(3)} auc
                    </span>
                  )}
                </div>
              ) : (
                <span className="text-[10px] text-[#5c5868] shrink-0">
                  {model.loaded ? 'No metrics' : 'Not trained'}
                </span>
              )}
            </div>
          )
        })}
      </div>

      {modelsData.models.every((m) => !m.loaded) && (
        <div className="mt-3 p-2 rounded border border-purple-400/30 bg-purple-400/5 text-[10px] text-purple-400">
          ⚠ No trained models found. Run <code className="bg-black/30 px-1 rounded">python train_models.py</code> in the backend to train. Demo predictions are active.
        </div>
      )}
    </div>
  )
}
