import ResultsClient from './ResultsClient';

export function generateStaticParams() {
  return [{ jobId: 'latest' }, { jobId: 'demo' }];
}

export default function Page() {
  return <ResultsClient />;
}
