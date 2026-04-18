import { useState, useRef, useEffect } from 'react'
import { ChevronDown, X, Shield, Swords } from 'lucide-react'

interface TeamSelectorProps {
  radiantTeams: string[]
  direTeams: string[]
  radiantTeam: string | null
  direTeam: string | null
  onRadiantChange: (team: string | null) => void
  onDireChange: (team: string | null) => void
}

function TeamCombobox({
  teams,
  value,
  onChange,
  placeholder,
  color,
  icon,
}: {
  teams: string[]
  value: string | null
  onChange: (v: string | null) => void
  placeholder: string
  color: 'radiant' | 'dire'
  icon: React.ReactNode
}) {
  const [open, setOpen]       = useState(false)
  const [query, setQuery]     = useState('')
  const containerRef          = useRef<HTMLDivElement>(null)

  const filtered = teams.filter(t =>
    t.toLowerCase().includes(query.toLowerCase()) && t !== '__other__' && t !== '__unknown__'
  )

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
        setQuery('')
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const borderActive = color === 'radiant' ? 'border-green-500/60' : 'border-red-500/60'
  const textColor    = color === 'radiant' ? 'text-green-400' : 'text-red-400'
  const bgHover      = color === 'radiant' ? 'hover:bg-green-500/10' : 'hover:bg-red-500/10'

  return (
    <div ref={containerRef} className="relative flex-1">
      {/* Trigger */}
      <div
        className={`flex items-center gap-2 px-3 py-2 rounded border cursor-pointer transition-colors
          ${open ? borderActive : 'border-[#2a2a3a] hover:border-[#3d3d55]'}
          bg-[#0d0d12]`}
        onClick={() => { setOpen(o => !o); setQuery('') }}
      >
        <span className={textColor}>{icon}</span>
        <span className={`flex-1 text-sm truncate ${value ? textColor : 'text-[#5c5868]'}`}>
          {value ?? placeholder}
        </span>
        {value ? (
          <button
            onClick={e => { e.stopPropagation(); onChange(null) }}
            className="text-[#5c5868] hover:text-[#e8e6f0]"
          >
            <X size={12} />
          </button>
        ) : (
          <ChevronDown size={13} className="text-[#5c5868]" />
        )}
      </div>

      {/* Dropdown */}
      {open && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 glass-card border border-[#2a2a3a] rounded overflow-hidden shadow-xl">
          <div className="p-2 border-b border-[#2a2a3a]">
            <input
              autoFocus
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search team..."
              className="w-full bg-[#0d0d12]/50 text-sm text-[#e8e6f0] placeholder:text-[#5c5868]
                px-2 py-1 rounded border border-[#2a2a3a] focus:outline-none focus:border-purple-500"
            />
          </div>
          <div className="max-h-48 overflow-y-auto">
            {filtered.length === 0 ? (
              <div className="px-3 py-2 text-xs text-[#5c5868]">No teams found</div>
            ) : (
              filtered.map(team => (
                <div
                  key={team}
                  className={`px-3 py-2 text-sm cursor-pointer transition-colors
                    ${team === value ? `${textColor} bg-[#2a2a3a]/40` : `text-[#9896a8] ${bgHover}`}`}
                  onClick={() => { onChange(team); setOpen(false); setQuery('') }}
                >
                  {team}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function TeamSelector({
  radiantTeams, direTeams,
  radiantTeam, direTeam,
  onRadiantChange, onDireChange,
}: TeamSelectorProps) {
  if (radiantTeams.length === 0 && direTeams.length === 0) return null

  return (
    <div className="glass-card p-3">
      <div className="section-title mb-2 flex items-center gap-1">
        Team Identity
        <span className="text-[9px] text-[#5c5868] normal-case tracking-normal ml-1">
          — helps the model factor in team win rates
        </span>
      </div>
      <div className="flex gap-3">
        <TeamCombobox
          teams={radiantTeams}
          value={radiantTeam}
          onChange={onRadiantChange}
          placeholder="Radiant team (optional)"
          color="radiant"
          icon={<Shield size={13} />}
        />
        <TeamCombobox
          teams={direTeams}
          value={direTeam}
          onChange={onDireChange}
          placeholder="Dire team (optional)"
          color="dire"
          icon={<Swords size={13} />}
        />
      </div>
    </div>
  )
}
