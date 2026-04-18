export interface Hero {
  id: number
  name: string
  roles: string[]
  attr: string
}

export interface Model {
  id: string
  display_name: string
  description: string
  loaded: boolean
  accuracy?: number
  auc?: number
}

export interface ModelsResponse {
  models: Model[]
  training_samples?: number
  feature_count?: number
}

export interface TeamsResponse {
  radiant_teams: string[]
  dire_teams: string[]
}

export interface Prediction {
  model_name: string
  display_name: string
  radiant_win_probability: number
  dire_win_probability: number
  predicted_winner: string
  confidence: string
  accuracy?: number
  auc?: number
  description: string
}

export interface PredictionResponse {
  radiant_heroes: number[]
  dire_heroes: number[]
  banned_heroes: number[]
  radiant_team?: string
  dire_team?: string
  predictions: Prediction[]
  ensemble_radiant_prob: number
  ensemble_winner: string
}

export interface Team {
  name: string
}