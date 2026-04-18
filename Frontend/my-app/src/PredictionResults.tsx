import { useEffect, useRef } from 'react'
import { Brain, TreePine, Zap, Trophy, TrendingUp, BarChart2, Leaf } from 'lucide-react'
import type { PredictionResponse } from '../types'

const MODEL_ICONS: Record<string, React.ReactNode> = {
  random_forest: <TreePine size={18} />,
  xgboost: <Zap size={18} />,
  neural_network: <Brain size={18} />,
  lightgbm: <Leaf size={18} />,
}

const MODEL_COLORS: Record<string, { primary: string; glow: string }> = {
  random_forest: { primary: '#4CAF50', glow: 'rgba(76,175,80,0.2)' },
  xgboost: { primary: '#FF9800', glow: 'rgba(255,152,0,0.2)' },
  neural_network: { primary: '#4a9eff', glow: 'rgba(74,158,255,0.2)' },
  lightgbm: { primary: '#9b59b6', glow: 'rgba(155,89,182,0.2)' },
}

const CONFIDENCE_COLORS: Record<string, string> = {
  High: 'text-purple-400 border-purple-400/40 bg-purple-400/10',
  Medium: 'text-purple-500 border-purple-500/40 bg-purple-500/10',
  Low: 'text-[#5c5868] border-[#2a2a3a] bg-[#2a2a3a]/20',
  'Toss-up': 'text-purple-400 border-purple-400/40 bg-purple-400/10',
}

interface ProbBarProps {
  radiantProb: number
  animate?: boolean
}

function ProbBar({ radiantProb, animate = true }: ProbBarProps) {
  const direProb = 1 - radiantProb
  const radiantPct = Math.round(radiantProb * 100)
  const direPct = 100 - radiantPct

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-green-400 font-semibold">{radiantPct}%</span>
        <span className="text-[#5c5868] text-[10px] uppercase tracking-widest">Win Probability</span>
        <span className="text-red-400 font-semibold">{direPct}%</span>
      </div>
      <div className="relative h-3 rounded-full overflow-hidden bg-[#0d0d12] border border-[#2a2a3a]">
        {/* Radiant bar */}
        <div
          className="absolute left-0 top-0 h-full transition-all duration-1000 ease-out"
          style={{
            width: animate ? `${radiantPct}%` : `${radiantPct}%`,
            background: 'linear-gradient(90deg, #7c3aed, #a855f7)',
          }}
        />
        {/* Center marker */}
        <div className="absolute left-1/2 top-0 h-full w-px bg-[#2a2a3a]/60 z-10" />
      </div>
      <div className="flex justify-between text-[9px] uppercase tracking-widest text-[#5c5868] mt-0.5">
        <span>◀ Radiant</span>
        <span>Dire ▶</span>
      </div>
    </div>
  )
}

interface PredictionResultsProps {
  result: PredictionResponse
}

export default function PredictionResults({ result }: PredictionResultsProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result])

  const ensemblePct = Math.round(result.ensemble_radiant_prob * 100)
  const isRadiantFavored = result.ensemble_radiant_prob > 0.5

  return (
    <div ref={containerRef} className="flex flex-col gap-4 animate-fade-up">
      {/* Ensemble Banner */}
      <div
        className={`glass-card p-4 relative overflow-hidden border
          ${isRadiantFavored ? 'border-green-500/40' : 'border-red-500/40'}`}
        style={{
          background: isRadiantFavored
            ? 'linear-gradient(135deg, rgba(124,58,237,0.05), rgba(15,18,25,0.95))'
            : 'linear-gradient(135deg, rgba(124,58,237,0.05), rgba(15,18,25,0.95))',
        }}
      >
        <div className="absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-20"
          style={{ background: isRadiantFavored ? '#a855f7' : '#c084fc' }} />

        <div className="flex items-center gap-3 mb-3">
          <Trophy size={16} className={isRadiantFavored ? 'text-green-400' : 'text-red-400'} />
          <span className="section-title">Ensemble Prediction</span>
          <span className="text-[#5c5868] text-xs">— Average of all 3 models</span>
        </div>

        <div className="flex items-end gap-4 mb-4">
          <div>
            <div className={`font-bold text-4xl
              ${isRadiantFavored ? 'text-green-400' : 'text-red-400'}`}>
              {isRadiantFavored ? ensemblePct : 100 - ensemblePct}%
            </div>
            <div className="text-lg text-[#e8e6f0]">
              {result.ensemble_winner} Victory
            </div>
          </div>
          <div className="flex-1">
            <ProbBar radiantProb={result.ensemble_radiant_prob} />
          </div>
        </div>
      </div>

      {/* Individual Model Cards */}
      <div className="section-title flex items-center gap-2">
        <BarChart2 size={12} />
        Individual Model Outputs
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {result.predictions.map((pred) => {
          const colors = MODEL_COLORS[pred.model_name] ?? MODEL_COLORS.xgboost
          const icon = MODEL_ICONS[pred.model_name]
          const isWinnerRadiant = pred.predicted_winner === 'Radiant'

          return (
            <div
              key={pred.model_name}
              className="glass-card p-4 flex flex-col gap-3 relative overflow-hidden"
              style={{ borderColor: `${colors.primary}33` }}
            >
              {/* Background glow */}
              <div
                className="absolute inset-0 opacity-5"
                style={{ background: `radial-gradient(ellipse at top right, ${colors.primary}, transparent 70%)` }}
              />

              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2" style={{ color: colors.primary }}>
                  {icon}
                  <span className="font-semibold text-base text-[#e8e6f0]">
                    {pred.display_name}
                  </span>
                </div>
                <span className={`stat-badge border ${CONFIDENCE_COLORS[pred.confidence]}`}>
                  {pred.confidence}
                </span>
              </div>

              {/* Winner */}
              <div className="flex items-center gap-2">
                <span className="text-[#5c5868] text-xs">Predicts:</span>
                <span
                  className={`font-bold text-xl
                    ${isWinnerRadiant ? 'text-green-400' : 'text-red-400'}`}
                >
                  {pred.predicted_winner}
                </span>
                <span className="text-xs text-[#5c5868] ml-auto">
                  {isWinnerRadiant
                    ? `${Math.round(pred.radiant_win_probability * 100)}%`
                    : `${Math.round(pred.dire_win_probability * 100)}%`}
                </span>
              </div>

              {/* Prob Bar */}
              <ProbBar radiantProb={pred.radiant_win_probability} />

              {/* Stats */}
              {(pred.accuracy !== null || pred.auc !== null) && (
                <div className="flex gap-2 pt-1 border-t border-[#2a2a3a]">
                  {pred.accuracy !== null && (
                    <div className="flex items-center gap-1">
                      <TrendingUp size={10} className="text-[#5c5868]" />
                      <span className="text-[10px] text-[#9896a8]">
                        ACC {(pred.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {pred.auc !== null && (
                    <div className="flex items-center gap-1">
                      <BarChart2 size={10} className="text-[#5c5868]" />
                      <span className="text-[10px] text-[#9896a8]">
                        AUC {pred.auc.toFixed(3)}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Description */}
              <p className="text-[10px] text-[#5c5868] leading-relaxed border-t border-[#2a2a3a] pt-2">
                {pred.description}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
