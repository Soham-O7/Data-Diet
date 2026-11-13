import ConfigureClient from './ConfigureClient';

export function generateStaticParams() {
  return [{ datasetId: 'mnist' }, { datasetId: 'iris' }, { datasetId: 'adult' }];
}

export default function Page() {
  return <ConfigureClient />;
}
