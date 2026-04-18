import { useState, useMemo } from 'react'
import { Search, X, Shield, Swords, Ban } from 'lucide-react'
import type { Hero, Team } from './types'

const ATTR_COLORS = { str: 'text-red-400', agi: 'text-green-400', int: 'text-blue-400' }
const ATTR_BG     = { str: 'bg-red-500/10', agi: 'bg-green-500/10', int: 'bg-blue-500/10' }
const ATTR_LABELS = { str: 'STR', agi: 'AGI', int: 'INT' }

type ActiveTarget = 'radiant' | 'dire' | 'ban'

interface HeroPickerProps {
  heroes: Hero[]
  radiantPicks: number[]
  direPicks: number[]
  bannedHeroes: number[]
  onPick:   (heroId: number, team: Team) => void
  onRemove: (heroId: number, team: Team) => void
}

function MiniSlot({
  heroId, team, heroes, onRemove,
}: { heroId: number | null; team: Team; heroes: Hero[]; onRemove: (id: number, t: Team) => void }) {
  const hero = heroId ? heroes.find(h => h.id === heroId) : null

  const colorClass =
    team === 'radiant' ? 'border-green-500/60 bg-green-500/10 text-green-400' :
    team === 'dire'    ? 'border-red-500/60 bg-red-500/10 text-red-400' :
                         'border-yellow-500/60 bg-yellow-500/10 text-yellow-400'

  const emptyClass =
    team === 'radiant' ? 'border-green-500/20' :
    team === 'dire'    ? 'border-red-500/20' :
                         'border-yellow-500/20'

  if (!hero) {
    return (
      <div className={`flex items-center justify-center rounded border-2 border-dashed h-12 w-10 ${emptyClass}`}>
        <span className="text-lg opacity-20">·</span>
      </div>
    )
  }

  const initials = hero.name.split(' ').map(w => w[0]).join('').slice(0, 3)

  return (
    <div
      className={`relative flex flex-col items-center justify-center rounded border h-12 w-10 px-0.5 cursor-pointer group ${colorClass}`}
      onClick={() => onRemove(hero.id, team)}
      title={`Remove ${hero.name}`}
    >
      <span className="text-[8px] font-bold text-center leading-tight">{initials}</span>
      <span className={`text-[7px] ${ATTR_COLORS[hero.attr]}`}>{ATTR_LABELS[hero.attr]}</span>
      <div className="absolute inset-0 rounded bg-black/70 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
        <X size={12} className="text-white" />
      </div>
    </div>
  )
}

function BanSlot({ heroId, heroes, onRemove }: { heroId: number | null; heroes: Hero[]; onRemove: (id: number) => void }) {
  const hero = heroId ? heroes.find(h => h.id === heroId) : null
  if (!hero) {
    return (
      <div className="flex items-center justify-center rounded border border-dashed border-yellow-500/15 h-9 w-9">
        <span className="text-sm opacity-20">·</span>
      </div>
    )
  }
  return (
    <div
      className="relative flex items-center justify-center rounded border border-yellow-500/50 bg-yellow-500/10 h-9 w-9 cursor-pointer group"
      onClick={() => onRemove(hero.id)}
      title={`Unban ${hero.name}`}
    >
      <span className="text-[8px] font-ui text-yellow-400 text-center leading-tight px-0.5">
        {hero.name.split(' ').map(w => w[0]).join('').slice(0, 3)}
      </span>
      <div className="absolute inset-0 rounded bg-black/70 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
        <X size={10} className="text-white" />
      </div>
    </div>
  )
}

export default function HeroPicker({
  heroes, radiantPicks, direPicks, bannedHeroes, onPick, onRemove,
}: HeroPickerProps) {
  const [search, setSearch]           = useState('')
  const [attrFilter, setAttrFilter]   = useState<'all' | 'str' | 'agi' | 'int'>('all')
  const [roleFilter, setRoleFilter]   = useState('all')
  const [target, setTarget]           = useState<ActiveTarget>('radiant')

  const allRoles = useMemo(() => {
    const s = new Set<string>()
    heroes.forEach(h => h.roles.forEach(r => s.add(r)))
    return ['all', ...Array.from(s).sort()]
  }, [heroes])

  const filtered = useMemo(() => heroes.filter(h => {
    const q    = h.name.toLowerCase().includes(search.toLowerCase())
    const attr = attrFilter === 'all' || h.attr === attrFilter
    const role = roleFilter === 'all' || h.roles.includes(roleFilter)
    return q && attr && role
  }), [heroes, search, attrFilter, roleFilter])

  const allUsed = new Set([...radiantPicks, ...direPicks, ...bannedHeroes])

  const getState = (id: number): ActiveTarget | null => {
    if (radiantPicks.includes(id)) return 'radiant'
    if (direPicks.includes(id))    return 'dire'
    if (bannedHeroes.includes(id)) return 'ban'
    return null
  }

  const handleHeroClick = (hero: Hero) => {
    const state = getState(hero.id)
    if (state !== null) {
      onRemove(hero.id, state)
      return
    }
    if (target === 'radiant' && radiantPicks.length < 5)  { onPick(hero.id, 'radiant'); return }
    if (target === 'dire'    && direPicks.length < 5)     { onPick(hero.id, 'dire');    return }
    if (target === 'ban'     && bannedHeroes.length < 10) { onPick(hero.id, 'ban');     return }
  }

  const targetFull =
    (target === 'radiant' && radiantPicks.length >= 5) ||
    (target === 'dire'    && direPicks.length >= 5)    ||
    (target === 'ban'     && bannedHeroes.length >= 10)

  return (
    <div className="flex flex-col gap-3">

      {/* ── Team Slots ── */}
      <div className="grid grid-cols-2 gap-3">
        {/* Radiant */}
        <div
          className={`glass-card p-3 cursor-pointer transition-all duration-150
            ${target === 'radiant' ? 'border-green-500/60 shadow-lg shadow-green-500/10' : 'hover:border-green-500/30'}`}
          onClick={() => setTarget('radiant')}
        >
          <div className="flex items-center gap-2 mb-2">
            <Shield size={13} className="text-green-400" />
            <span className="section-title text-green-400">Radiant {radiantPicks.length}/5</span>
            {target === 'radiant' && (
              <span className="ml-auto text-[9px] text-green-400/60 animate-pulse">● active</span>
            )}
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {Array.from({ length: 5 }).map((_, i) => (
              <MiniSlot key={i} heroId={radiantPicks[i] ?? null} team="radiant" heroes={heroes} onRemove={(id, t) => onRemove(id, t)} />
            ))}
          </div>
        </div>

        {/* Dire */}
        <div
          className={`glass-card p-3 cursor-pointer transition-all duration-150
            ${target === 'dire' ? 'border-red-500/60 shadow-lg shadow-red-500/10' : 'hover:border-red-500/30'}`}
          onClick={() => setTarget('dire')}
        >
          <div className="flex items-center gap-2 mb-2">
            <Swords size={13} className="text-red-400" />
            <span className="section-title text-red-400">Dire {direPicks.length}/5</span>
            {target === 'dire' && (
              <span className="ml-auto text-[9px] text-red-400/60 animate-pulse">● active</span>
            )}
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {Array.from({ length: 5 }).map((_, i) => (
              <MiniSlot key={i} heroId={direPicks[i] ?? null} team="dire" heroes={heroes} onRemove={(id, t) => onRemove(id, t)} />
            ))}
          </div>
        </div>
      </div>

      {/* ── Ban Slots ── */}
      <div
        className={`glass-card p-3 cursor-pointer transition-all duration-150
          ${target === 'ban' ? 'border-yellow-500/50 shadow-lg shadow-yellow-500/5' : 'hover:border-yellow-500/20'}`}
        onClick={() => setTarget('ban')}
      >
        <div className="flex items-center gap-2 mb-2">
          <Ban size={13} className="text-yellow-400" />
          <span className="section-title text-yellow-400">Bans {bannedHeroes.length}/10</span>
          <span className="text-[9px] text-[#5c5868] ml-1">(optional — improves prediction if models were trained with bans)</span>
          {target === 'ban' && (
            <span className="ml-auto text-[9px] text-yellow-400/60 animate-pulse">● active</span>
          )}
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {Array.from({ length: 10 }).map((_, i) => (
            <BanSlot key={i} heroId={bannedHeroes[i] ?? null} heroes={heroes} onRemove={id => onRemove(id, 'ban')} />
          ))}
        </div>
      </div>

      {/* ── Target switcher hint ── */}
      <div className="flex items-center gap-2 px-1">
        <span className="text-[10px] text-[#5c5868]">Adding to:</span>
        {(['radiant', 'dire', 'ban'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTarget(t)}
            className={`px-2 py-0.5 rounded text-[10px] uppercase transition-colors border
              ${target === t
                ? t === 'radiant' ? 'bg-green-500/20 border-green-500 text-green-400'
                  : t === 'dire'  ? 'bg-red-500/20 border-red-500 text-red-400'
                  : 'bg-yellow-500/20 border-yellow-500 text-yellow-400'
                : 'border-[#2a2a3a] text-[#5c5868] hover:border-[#3d3d55]'}`}
          >
            {t}
          </button>
        ))}
        {targetFull && (
          <span className="text-[10px] text-purple-400 ml-1">
            {target} is full
          </span>
        )}
      </div>

      {/* ── Search & Filters ── */}
      <div className="glass-card p-3 flex flex-col gap-2">
        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#5c5868]" />
          <input
            type="text"
            placeholder="Search hero..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full bg-[#0d0d12] border border-[#2a2a3a] rounded pl-8 pr-3 py-1.5 text-sm
              text-[#e8e6f0] placeholder:text-[#5c5868] focus:outline-none focus:border-purple-500"
          />
          {search && (
            <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#5c5868] hover:text-[#e8e6f0]">
              <X size={12} />
            </button>
          )}
        </div>

        <div className="flex gap-1.5 flex-wrap items-center">
          {(['all', 'str', 'agi', 'int'] as const).map(a => (
            <button
              key={a}
              onClick={() => setAttrFilter(a)}
              className={`px-2 py-0.5 rounded text-xs uppercase transition-colors border
                ${attrFilter === a
                  ? a === 'str' ? 'bg-red-500/20 text-red-400 border-red-500/50'
                    : a === 'agi' ? 'bg-green-500/20 text-green-400 border-green-500/50'
                    : a === 'int' ? 'bg-blue-500/20 text-blue-400 border-blue-500/50'
                    : 'bg-purple-500/20 text-purple-400 border-purple-500/50'
                  : 'text-[#5c5868] border-[#2a2a3a] hover:border-[#3d3d55]'}`}
            >
              {a === 'all' ? 'All' : ATTR_LABELS[a]}
            </button>
          ))}
          <div className="w-px h-4 bg-[#2a2a3a]" />
          <select
            value={roleFilter}
            onChange={e => setRoleFilter(e.target.value)}
            className="bg-[#0d0d12] border border-[#2a2a3a] rounded px-2 py-0.5 text-xs
              text-[#9896a8] focus:outline-none focus:border-purple-500"
          >
            {allRoles.map(r => (
              <option key={r} value={r}>{r === 'all' ? 'All Roles' : r}</option>
            ))}
          </select>
          <span className="text-[10px] text-[#5c5868] ml-auto">{filtered.length} shown</span>
        </div>
      </div>

      {/* ── Hero Grid ── */}
      <div className="glass-card p-3">
        <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 xl:grid-cols-14 gap-1 max-h-64 overflow-y-auto pr-1">
          {filtered.map(hero => {
            const state = getState(hero.id)
            const used  = state !== null

            const borderClass =
              state === 'radiant' ? 'border-green-500 bg-green-500/15 text-green-400' :
              state === 'dire'    ? 'border-red-500 bg-red-500/15 text-red-400' :
              state === 'ban'     ? 'border-yellow-500/60 bg-yellow-500/10 text-yellow-400 opacity-60' :
              targetFull          ? 'border-[#2a2a3a]/40 text-[#5c5868]/40 cursor-not-allowed' :
                                    `border-[#2a2a3a] hover:border-purple-500 text-[#9896a8] hover:text-[#e8e6f0] ${ATTR_BG[hero.attr]}`

            return (
              <button
                key={hero.id}
                onClick={() => handleHeroClick(hero)}
                title={`${hero.name} — ${hero.roles.join(', ')}`}
                disabled={!used && targetFull}
                className={`relative flex flex-col items-center justify-center p-1 rounded border
                  text-center transition-all duration-100 cursor-pointer group ${borderClass}`}
              >
                <span className={`text-[7px] font-semibold ${ATTR_COLORS[hero.attr]}`}>
                  {ATTR_LABELS[hero.attr]}
                </span>
                <span className="text-[8px] leading-tight text-center mt-0.5 line-clamp-2 w-full">
                  {hero.name}
                </span>
                {used && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full flex items-center justify-center
                    bg-[#0d0d12] border border-current z-10">
                    <X size={6} />
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
