import { useState, useEffect, useCallback } from 'react'
import { Swords, RotateCcw, ChevronRight, Wifi, WifiOff } from 'lucide-react'
import toast, { Toaster } from 'react-hot-toast'
import HeroPicker from './HeroPicker'
import PredictionResults from './PredictionResults'
import ModelInfoPanel from './ModelInfoPanel'
import TeamSelector from './TeamSelector'
import { fetchHeroes, fetchModels, fetchTeams, predict, healthCheck } from './utils/api'
import type { Hero, PredictionResponse, ModelsResponse, TeamsResponse, Team } from './types'
import { Analytics } from '@vercel/analytics/react';

export default function App() {
  const [heroes, setHeroes]           = useState<Hero[]>([])
  const [modelsData, setModelsData]   = useState<ModelsResponse | null>(null)
  const [teamsData, setTeamsData]     = useState<TeamsResponse | null>(null)

  const [radiantPicks, setRadiantPicks] = useState<number[]>([])
  const [direPicks, setDirePicks]       = useState<number[]>([])
  const [bannedHeroes, setBannedHeroes] = useState<number[]>([])
  const [radiantTeam, setRadiantTeam]   = useState<string | null>(null)
  const [direTeam, setDireTeam]         = useState<string | null>(null)

  const [result, setResult]           = useState<PredictionResponse | null>(null)
  const [loading, setLoading]         = useState(false)
  const [apiOnline, setApiOnline]     = useState<boolean | null>(null)
  const [heroesLoading, setHeroesLoading] = useState(true)

  useEffect(() => {
    const init = async () => {
      const online = await healthCheck()
      setApiOnline(online)
      if (online) {
        try {
          const [heroData, modelData, teamData] = await Promise.all([
            fetchHeroes(), fetchModels(), fetchTeams(),
          ])
          setHeroes(heroData)
          setModelsData(modelData)
          setTeamsData(teamData)
        } catch {
          toast.error('Failed to load data from API')
        }
      } else {
        toast.error('Backend offline — start FastAPI on port 8000', { duration: 6000 })
      }
      setHeroesLoading(false)
    }
    init()
  }, [])

  const handlePick = useCallback((heroId: number, team: Team) => {
    if (team === 'radiant') setRadiantPicks(p => p.length < 5  ? [...p, heroId] : p)
    else if (team === 'dire') setDirePicks(p => p.length < 5   ? [...p, heroId] : p)
    else if (team === 'ban')  setBannedHeroes(p => p.length < 10 ? [...p, heroId] : p)
    setResult(null)
  }, [])

  const handleRemove = useCallback((heroId: number, team: Team) => {
    if (team === 'radiant') setRadiantPicks(p => p.filter(id => id !== heroId))
    else if (team === 'dire') setDirePicks(p => p.filter(id => id !== heroId))
    else if (team === 'ban')  setBannedHeroes(p => p.filter(id => id !== heroId))
    setResult(null)
  }, [])

  const handleReset = () => {
    setRadiantPicks([]); setDirePicks([]); setBannedHeroes([])
    setRadiantTeam(null); setDireTeam(null); setResult(null)
  }

  const handlePredict = async () => {
    if (radiantPicks.length !== 5 || direPicks.length !== 5) {
      toast.error('Pick 5 heroes for each team')
      return
    }
    setLoading(true)
    try {
      const res = await predict(radiantPicks, direPicks, bannedHeroes, radiantTeam, direTeam)
      setResult(res)
      toast.success('Prediction ready!')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const readyToPredict = radiantPicks.length === 5 && direPicks.length === 5

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0d0d12] via-[#1a1025] to-[#0d0d12]">
      {/* Ambient BG */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-[120px] opacity-20 bg-purple-600" />
        <div className="absolute top-0 right-1/4 w-96 h-96 rounded-full blur-[120px] opacity-20 bg-purple-500" />
      </div>

      {/* Header */}
      <header className="relative border-b border-[#2a2a3a] bg-[#14141a]/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <Analytics />
          <div className="flex items-center gap-3">
            <div className="relative">
              <Swords size={22} className="text-purple-400" />
              <div className="absolute inset-0 blur-md bg-purple-400 opacity-30" />
            </div>
            <div>
              <h1 className="font-bold text-xl text-[#e8e6f0] tracking-wide">
                RobotSkie <span className="text-purple-400">V2</span>
              </h1>
              <div className="text-[10px] text-[#5c5868] tracking-widest uppercase">
                Pro Match Predictor · Picks + Bans + Team Identity
              </div>
            </div>
          </div>
          <div className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded border
            ${apiOnline === true  ? 'text-green-400 border-green-400/30 bg-green-400/5'
            : apiOnline === false ? 'text-red-400 border-red-400/30 bg-red-400/5'
            :                       'text-[#5c5868] border-[#2a2a3a]'}`}>
            {apiOnline === true ? <Wifi size={11} /> : apiOnline === false ? <WifiOff size={11} /> : null}
            {apiOnline === null ? 'Connecting...' : apiOnline ? 'API Online' : 'API Offline'}
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-4 py-6 flex flex-col gap-6">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

          {/* Left: Draft + Teams */}
          <div className="xl:col-span-2 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-lg text-[#e8e6f0] flex items-center gap-2">
                <span className="w-1 h-5 bg-purple-500 rounded-full" />
                Draft
              </h2>
              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 text-xs text-[#5c5868] hover:text-[#e8e6f0]
                  border border-[#2a2a3a] hover:border-[#3d3d55] px-2.5 py-1 rounded transition-colors"
              >
                <RotateCcw size={11} /> Reset
              </button>
            </div>

            {heroesLoading ? (
              <div className="glass-card p-8 text-center text-[#5c5868] text-sm animate-pulse">
                Loading heroes...
              </div>
            ) : heroes.length === 0 ? (
              <div className="glass-card p-8 text-center">
                <p className="text-[#5c5868] text-sm mb-2">Could not load heroes from API.</p>
                <p className="text-[#3d3d55] text-xs">Make sure FastAPI is running on port 8000.</p>
              </div>
            ) : (
              <HeroPicker
                heroes={heroes}
                radiantPicks={radiantPicks}
                direPicks={direPicks}
                bannedHeroes={bannedHeroes}
                onPick={handlePick}
                onRemove={handleRemove}
              />
            )}

            {/* Team selector — shown when teams are available from training */}
            {teamsData && (teamsData.radiant_teams.length > 0 || teamsData.dire_teams.length > 0) && (
              <TeamSelector
                radiantTeams={teamsData.radiant_teams}
                direTeams={teamsData.dire_teams}
                radiantTeam={radiantTeam}
                direTeam={direTeam}
                onRadiantChange={setRadiantTeam}
                onDireChange={setDireTeam}
              />
            )}
          </div>

          {/* Right sidebar */}
          <div className="xl:col-span-1 flex flex-col gap-4">
            <div>
              <h2 className="font-semibold text-lg text-[#e8e6f0] flex items-center gap-2 mb-3">
                <span className="w-1 h-5 bg-purple-400 rounded-full" />
                Models
              </h2>
              <ModelInfoPanel modelsData={modelsData} />
            </div>

            <div className="glass-card p-4">
              <div className="section-title mb-3">How It Works</div>
              <ol className="flex flex-col gap-2">
                {[
                  'Click a team slot header (Radiant / Dire / Ban) to set the active target',
                  'Click heroes in the grid to add them to the active target',
                  'Optionally select team names if models were trained with team features',
                  'Click Predict — all 4 models run and give independent probabilities',
                ].map((step, i) => (
                  <li key={i} className="flex items-start gap-2 text-[11px] text-[#5c5868]">
                    <span className="text-purple-400 font-bold shrink-0">{i + 1}.</span>
                    {step}
                  </li>
                ))}
              </ol>
            </div>

            <div className="glass-card p-4 flex flex-col items-center gap-2">
              <div className="section-title">Support the Project</div>
              <div className="w-32 h-32 rounded-lg border border-[#2a2a3a] bg-[#0d0d12] flex items-center justify-center overflow-hidden">
                <img 
                  src="/qr-code.png" 
                  alt="Donation QR Code" 
                  className="w-full h-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none'
                  }}
                />
              </div>
              <span className="text-[10px] text-[#5c5868]">Scan to donate</span>
            </div>

            <div className="glass-card p-4">
              <div className="section-title mb-2">Feature Set</div>
              <div className="flex flex-col gap-1.5 text-[11px]">
                {[
                  { label: 'Hero picks', desc: '+1 Radiant / −1 Dire per hero', color: 'text-purple-400' },
                  { label: 'Hero bans',  desc: '+1 if hero was banned',          color: 'text-yellow-400' },
                  { label: 'Team ID',    desc: 'One-hot team name encoding',     color: 'text-purple-300'  },
                ].map(f => (
                  <div key={f.label} className="flex items-start gap-2">
                    <span className={`font-semibold w-20 shrink-0 ${f.color}`}>{f.label}</span>
                    <span className="text-[#5c5868]">{f.desc}</span>
                  </div>
                ))}
                {modelsData?.feature_count ? (
                  <div className="mt-1 pt-1 border-t border-[#2a2a3a] text-[#5c5868]">
                    Total: <span className="text-[#e8e6f0]">{modelsData.feature_count}</span> features per match
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>

        {/* Predict Button */}
        <div className="flex flex-col items-center gap-3">
          <button
            onClick={handlePredict}
            disabled={!readyToPredict || loading}
            className={`relative group flex items-center gap-3 px-10 py-4 rounded font-bold text-lg
              uppercase tracking-widest transition-all duration-200
              ${readyToPredict && !loading
                ? 'bg-purple-600 text-white hover:bg-purple-500 shadow-lg shadow-purple-600/20 hover:shadow-purple-500/40 cursor-pointer'
                : 'bg-[#2a2a3a] text-[#5c5868] cursor-not-allowed opacity-60'}`}
          >
            {loading ? (
              <>
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Running 3 Models...
              </>
            ) : (
              <>
                <Swords size={18} />
                Predict Match
                <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </>
            )}
          </button>

          {/* Progress dots */}
          {!readyToPredict && (
            <div className="flex items-center gap-4 text-xs text-[#5c5868]">
              {[
                { label: 'Radiant', count: radiantPicks.length, color: 'bg-green-500' },
                { label: 'Dire',    count: direPicks.length,    color: 'bg-red-500'    },
              ].map(({ label, count, color }) => (
                <div key={label} className="flex items-center gap-2">
                  <div className="flex gap-0.5">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className={`w-2 h-2 rounded-full transition-colors
                        ${i < count ? color : 'bg-[#2a2a3a]'}`} />
                    ))}
                  </div>
                  <span className={count === 5 ? 'text-[#e8e6f0]' : ''}>{label} {count}/5</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Results */}
        {result && (
          <div>
            <h2 className="font-semibold text-lg text-[#e8e6f0] flex items-center gap-2 mb-3">
              <span className="w-1 h-5 bg-red-500 rounded-full" />
              Prediction Results
              {result.radiant_team && result.dire_team && (
                <span className="text-sm text-[#5c5868] font-normal ml-1">
                  — {result.radiant_team} vs {result.dire_team}
                </span>
              )}
            </h2>
            <PredictionResults result={result} />
          </div>
        )}
      </main>

      <footer className="border-t border-[#2a2a3a] mt-12 py-4">
        <div className="max-w-7xl mx-auto px-4 flex items-center justify-between text-[10px] text-[#5c5868]">
          <span>Dota 2 Oracle · Picks + Bans + Teams</span>
          <span>Random Forest · XGBoost · Neural Network</span>
        </div>
      </footer>

      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#14141a', color: '#e8e6f0',
            border: '1px solid #2a2a3a',
            fontSize: '12px',
          },
        }}
      />
    </div>
  )
}
