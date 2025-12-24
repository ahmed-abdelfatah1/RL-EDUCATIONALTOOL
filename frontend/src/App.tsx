import { useEffect, useState, useCallback, useMemo } from "react"
import { fetchEnvironments, EnvironmentInfo } from "./api/environmentApi"
import { fetchAlgorithms, AlgorithmInfo } from "./api/algorithmApi"
import { TrainingRunResponse } from "./api/trainingApi"
import { resetEnvironment, stepEnvironment } from "./api/environmentStateApi"
import { RunControls } from "./components/controls/RunControls"
import { RewardChart } from "./components/visualization/RewardChart"
import { EnvironmentView } from "./components/visualization/EnvironmentView"
import { useEnvironmentStream } from "./hooks/useEnvironmentStream"
import type { AnyEnvRenderState } from "./types/environment"
import { Button } from "./components/ui/button"
import { Card } from "./components/ui/card"
import { Label } from "./components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Switch } from "./components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs"
import {
  Brain,
  Grid3x3,
  Play,
  RotateCcw,
  Eye,
  Target,
  TrendingUp,
  BookOpen,
} from "lucide-react"
import { AlgorithmExplanation } from "./components/education/AlgorithmExplanation"
import { ValueFunctionView } from "./components/visualization/ValueFunctionView"
import { PolicyView } from "./components/visualization/PolicyView"

function App() {
  const [environments, setEnvironments] = useState<EnvironmentInfo[]>([])
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([])
  const [selectedEnv, setSelectedEnv] = useState<string>("")
  const [selectedAlgo, setSelectedAlgo] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [trainingResult, setTrainingResult] = useState<TrainingRunResponse | null>(null)
  const [envState, setEnvState] = useState<AnyEnvRenderState | null>(null)
  const [stepping, setStepping] = useState(false)
  const [followTraining, setFollowTraining] = useState(true)

  // Filter algorithms based on selected environment compatibility
  const compatibleAlgorithms = useMemo(() => {
    if (!selectedEnv) return algorithms
    return algorithms.filter((algo) => algo.supports_envs.includes(selectedEnv))
  }, [algorithms, selectedEnv])

  useEffect(() => {
    async function loadData() {
      try {
        const [envs, algos] = await Promise.all([
          fetchEnvironments(),
          fetchAlgorithms(),
        ])

        setEnvironments(envs)
        if (envs.length > 0) {
          setSelectedEnv(envs[0].name)
        }

        setAlgorithms(algos)
        if (algos.length > 0) {
          setSelectedAlgo(algos[0].name)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data")
      } finally {
        setLoading(false)
      }
    }

    void loadData()
  }, [])

  // Auto-select first compatible algorithm when environment changes
  useEffect(() => {
    if (!selectedEnv || algorithms.length === 0) return

    const compatible = algorithms.filter((algo) =>
      algo.supports_envs.includes(selectedEnv)
    )

    if (compatible.length > 0) {
      const currentIsCompatible = compatible.some((a) => a.name === selectedAlgo)
      if (!currentIsCompatible) {
        setSelectedAlgo(compatible[0].name)
      }
    }
  }, [selectedEnv, algorithms, selectedAlgo])

  useEffect(() => {
    if (!selectedEnv) return

    // Clear previous results when environment changes
    setTrainingResult(null)

    void (async () => {
      try {
        const snapshot = await resetEnvironment(selectedEnv)
        setEnvState(snapshot.state)
      } catch (err) {
        console.error("Failed to reset environment:", err)
        setEnvState(null)
      }
    })()
  }, [selectedEnv])

  // Clear results when algorithm changes
  useEffect(() => {
    setTrainingResult(null)
  }, [selectedAlgo])

  const handleStreamState = useCallback((state: AnyEnvRenderState) => {
    setEnvState(state)
  }, [])

  useEnvironmentStream({
    envName: selectedEnv || null,
    enabled: followTraining,
    onState: handleStreamState,
  })

  const getDefaultAction = (envName: string): number => {
    switch (envName) {
      case "mountaincar":
        return 2
      case "cartpole":
        return 1
      case "gridworld":
      case "frozenlake":
        return 1
      default:
        return 0
    }
  }

  async function handleStep(action?: number) {
    if (!selectedEnv || stepping) return

    setStepping(true)
    try {
      const actionToTake = action ?? getDefaultAction(selectedEnv)
      const snapshot = await stepEnvironment(selectedEnv, actionToTake)
      setEnvState(snapshot.state)
    } catch (err) {
      console.error("Failed to step environment:", err)
    } finally {
      setStepping(false)
    }
  }

  async function handleReset() {
    if (!selectedEnv) return

    try {
      const snapshot = await resetEnvironment(selectedEnv)
      setEnvState(snapshot.state)
    } catch (err) {
      console.error("Failed to reset environment:", err)
    }
  }

  const selectedEnvInfo = environments.find((e) => e.name === selectedEnv)
  const selectedAlgoInfo = algorithms.find((a) => a.name === selectedAlgo)

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-foreground">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-6">
        <Card className="border-destructive/50 bg-card/50 backdrop-blur-sm max-w-md w-full">
          <div className="p-6 space-y-4">
            <h2 className="text-lg font-semibold text-destructive">Error</h2>
            <p className="text-sm text-foreground">{error}</p>
            <p className="text-xs text-muted-foreground">
              Make sure the backend is running on localhost:8000
            </p>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="h-6 w-6 text-primary" />
              <div>
                <h1 className="text-xl font-bold text-foreground">RL Education Tool</h1>
                <p className="text-xs text-muted-foreground">Reinforcement Learning Playground</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
          <div className="space-y-6">
            <Card className="border-border bg-card/50 backdrop-blur-sm">
              <div className="p-6 space-y-6">
                <div className="flex items-center gap-2 pb-4 border-b border-border">
                  <Grid3x3 className="h-5 w-5 text-blue-500" />
                  <h2 className="text-lg font-semibold text-foreground">Configuration</h2>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-foreground flex items-center gap-2">
                    <Grid3x3 className="h-4 w-4 text-muted-foreground" />
                    Environment
                  </Label>
                  <Select value={selectedEnv} onValueChange={setSelectedEnv}>
                    <SelectTrigger className="w-full bg-background border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {environments.map((env) => (
                        <SelectItem key={env.name} value={env.name}>
                          {env.display_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedEnvInfo && (
                    <p className="text-xs text-muted-foreground">{selectedEnvInfo.description}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label className="text-sm font-medium text-foreground flex items-center gap-2">
                    <Brain className="h-4 w-4 text-muted-foreground" />
                    Algorithm
                    <span className="text-xs text-muted-foreground font-normal">
                      ({compatibleAlgorithms.length} compatible)
                    </span>
                  </Label>
                  <Select value={selectedAlgo} onValueChange={setSelectedAlgo}>
                    <SelectTrigger className="w-full bg-background border-border">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {compatibleAlgorithms.length > 0 ? (
                        compatibleAlgorithms.map((algo) => (
                          <SelectItem key={algo.name} value={algo.name}>
                            {algo.display_name}
                          </SelectItem>
                        ))
                      ) : (
                        <div className="px-2 py-1.5 text-sm text-muted-foreground">
                          No compatible algorithms
                        </div>
                      )}
                    </SelectContent>
                  </Select>
                  {selectedAlgoInfo && (
                    <p className="text-xs text-muted-foreground">{selectedAlgoInfo.description}</p>
                  )}
                </div>


                <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center gap-2">
                    <Eye className="h-4 w-4 text-muted-foreground" />
                    <Label htmlFor="follow-training" className="text-sm font-medium cursor-pointer">
                      Follow Training
                    </Label>
                  </div>
                  <div className="flex items-center gap-2">
                    {followTraining && (
                      <span className="flex h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                    )}
                    <Switch
                      id="follow-training"
                      checked={followTraining}
                      onCheckedChange={setFollowTraining}
                    />
                  </div>
                </div>
              </div>
            </Card>

            <Card className="border-border bg-card/50 backdrop-blur-sm">
              <div className="p-6 space-y-4">
                <div className="flex items-center gap-2 pb-4 border-b border-border">
                  <Target className="h-5 w-5 text-cyan-500" />
                  <h2 className="text-lg font-semibold text-foreground">Manual Control</h2>
                </div>
                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    className="flex-1 gap-2 bg-transparent"
                    onClick={() => void handleStep()}
                    disabled={stepping}
                  >
                    <Play className="h-4 w-4" />
                    Step
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1 gap-2 bg-transparent"
                    onClick={() => void handleReset()}
                  >
                    <RotateCcw className="h-4 w-4" />
                    Reset
                  </Button>
                </div>
              </div>
            </Card>

            <RunControls
              selectedEnv={selectedEnv}
              selectedAlgo={selectedAlgo}
              onResult={setTrainingResult}
            />
          </div>

          <div className="space-y-6">
            <Tabs defaultValue="environment" className="w-full">
              <TabsList className="grid w-full grid-cols-3 bg-muted/50">
                <TabsTrigger value="environment" className="gap-2">
                  <Grid3x3 className="h-4 w-4" />
                  Environment
                </TabsTrigger>
                <TabsTrigger value="metrics" className="gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Metrics
                </TabsTrigger>
                <TabsTrigger value="learn" className="gap-2">
                  <BookOpen className="h-4 w-4" />
                  Learn
                </TabsTrigger>
              </TabsList>

              <TabsContent value="environment" className="mt-6">
                <EnvironmentView envName={selectedEnv} state={envState} />
              </TabsContent>

              <TabsContent value="metrics" className="mt-6">
                <RewardChart episodes={trainingResult?.episodes ?? []} />
              </TabsContent>

              <TabsContent value="learn" className="mt-6 h-[600px] space-y-6 overflow-y-auto pr-2">
                <AlgorithmExplanation algorithmName={selectedAlgo} />

                {trainingResult?.value_function && (
                  <div className="h-[400px]">
                    <ValueFunctionView
                      envName={selectedEnv}
                      valueFunction={trainingResult.value_function}
                    />
                  </div>
                )}

                {trainingResult?.policy && (
                  <div className="h-[400px]">
                    <PolicyView
                      envName={selectedEnv}
                      policy={trainingResult.policy}
                    />
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
