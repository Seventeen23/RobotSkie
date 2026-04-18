const BASE_URL = "http://localhost:8000"

const handleResponse = async (res: Response) => {
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    console.error(`API Error ${res.status}:`, error)
    throw new Error(error.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── API calls ─────────────────────────────
export const fetchHeroes = async () => {
  const res = await fetch(`${BASE_URL}/heroes`)
  const data = await handleResponse(res)
  return data.heroes
}

export const fetchModels = async () => {
  const res = await fetch(`${BASE_URL}/models`)
  return await handleResponse(res)
}

export const fetchTeams = async () => {
  const res = await fetch(`${BASE_URL}/teams`)
  return await handleResponse(res)
}

export const healthCheck = async () => {
  const res = await fetch(`${BASE_URL}/health`)
  const data = await handleResponse(res)
  return data.status === "ok"
}

export const predict = async (
  radiant_heroes: number[],
  dire_heroes: number[],
  banned_heroes: number[] = [],
  radiant_team: string | null = null,
  dire_team: string | null = null
) => {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      radiant_heroes,
      dire_heroes,
      banned_heroes,
      radiant_team,
      dire_team,
    })
  })

  return handleResponse(res)
}