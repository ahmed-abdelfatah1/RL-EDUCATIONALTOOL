import { useState } from "react"
import { runTraining, TrainingRunResponse } from "../../api/trainingApi"
import { Button } from "../ui/button"
import { Card } from "../ui/card"
import { Input } from "../ui/input"
import { Label } from "../ui/label"
import { Settings, Zap, Activity } from "lucide-react"

interface RunControlsProps {
  selectedEnv: string
  selectedAlgo: string
  onResult: (result: TrainingRunResponse) => void
}

export function RunControls({
  selectedEnv,
  selectedAlgo,
  onResult,
}: RunControlsProps) {
  const [numEpisodes, setNumEpisodes] = useState(10)
  const [maxStepsPerEpisode, setMaxStepsPerEpisode] = useState(50)
  const [discountFactor, setDiscountFactor] = useState(0.99)
  const [learningRate, setLearningRate] = useState(0.1)
  const [epsilon, setEpsilon] = useState(0.1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleRun() {
    setLoading(true)
    setError(null)
    try {
      const result = await runTraining({
        env_name: selectedEnv,
        algorithm_name: selectedAlgo,
        num_episodes: numEpisodes,
        max_steps_per_episode: maxStepsPerEpisode,
        discount_factor: discountFactor,
        learning_rate: learningRate,
        epsilon,
      })
      onResult(result)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="border-border bg-card/50 backdrop-blur-sm">
      <div className="p-6 space-y-6">
        <div className="flex items-center gap-2 pb-4 border-b border-border">
          <Settings className="h-5 w-5 text-purple-500" />
          <h2 className="text-lg font-semibold text-foreground">Training Parameters</h2>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="episodes" className="text-sm text-muted-foreground">
              Episodes
            </Label>
            <Input
              id="episodes"
              type="number"
              value={numEpisodes}
              onChange={(e) => setNumEpisodes(Number(e.target.value))}
              className="bg-background border-border"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="max-steps" className="text-sm text-muted-foreground">
              Max steps / episode
            </Label>
            <Input
              id="max-steps"
              type="number"
              value={maxStepsPerEpisode}
              onChange={(e) => setMaxStepsPerEpisode(Number(e.target.value))}
              className="bg-background border-border"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="discount" className="text-sm text-muted-foreground">
                Discount (γ)
              </Label>
              <Input
                id="discount"
                type="number"
                step="0.01"
                value={discountFactor}
                onChange={(e) => setDiscountFactor(Number(e.target.value))}
                className="bg-background border-border"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="learning-rate" className="text-sm text-muted-foreground">
                Learning rate (α)
              </Label>
              <Input
                id="learning-rate"
                type="number"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="bg-background border-border"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="epsilon" className="text-sm text-muted-foreground">
              Epsilon (ε)
            </Label>
            <Input
              id="epsilon"
              type="number"
              step="0.01"
              value={epsilon}
              onChange={(e) => setEpsilon(Number(e.target.value))}
              className="bg-background border-border"
            />
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <p className="text-sm text-destructive-foreground">{error}</p>
          </div>
        )}

        <Button
          onClick={() => void handleRun()}
          disabled={loading}
          className="w-full gap-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
        >
          {loading ? (
            <>
              <Activity className="h-4 w-4 animate-pulse" />
              Training...
            </>
          ) : (
            <>
              <Zap className="h-4 w-4" />
              Run Training
            </>
          )}
        </Button>
      </div>
    </Card>
  )
}
